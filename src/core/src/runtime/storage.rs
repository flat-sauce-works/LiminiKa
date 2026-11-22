// name: src/core/src/runtime/storage.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-Copy In-Memory Mapped Storage (`MappedStorageEngine` 64B `Pod`,
//! [`LmnkContainerHeader`] 64B + [`LmnkSectionEntry`] 16B unified `.lmnk` container zero-copy
//! persistence & evolution ([`LmnkContainerView`]), [`KvKeyframeCache`], [`MappedStorage`],
//! phase packet codecs, and [`KvKeyframePacket`]).

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use core::mem::{align_of, offset_of, size_of};

pub use crate::foundation::KvMotionDeltaPacket;
use crate::foundation::{
    apply_pframe_motion_step_words_64, branchless_abs_f32, branchless_atan2_f32,
    branchless_clamp_f32, branchless_copysign_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_q7_accumulate_64, branchless_q7_inhibit_words_64,
    branchless_q7_phase_step_filter_64, branchless_relu_f32, branchless_sincos_q7,
    branchless_sqrt_f32, branchless_von_mises_kappa, circular_r_bar_from_words_64,
    circular_resultant_from_words_64, coherence_from_diff_words_64,
    compute_aligned64_payload_crc32_ieee, compute_crc32_ieee, crc32_block64_zero_seed_words,
    encode_pframe_motion_delta_64, finalize_pframe_chain_words_64, load_q7_words_64,
    pack_action_slot_tag, seek_pframe_chain_words_64, store_q7_words_64, swar_q7_add_sat_x8,
    swar_q7_circular_add_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_64, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, AlignedSlice, AlignedSlice16, AlignedSlice32, AlignedSlice64,
    AlignedSliceMut64, Pod, TaggedPtr, ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, LmnkResult, PhaseQ7, RuntimeConfig,
    DEFAULT_ENTROPY_EPS, LMNK_ABI_VERSION_HEX, LMNK_ERR_BUFFER_TOO_SMALL,
    LMNK_ERR_CONTAINER_CORRUPTED, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_INVALID_STATE,
    LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_ERR_VERSION_MISMATCH, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use crate::runtime::gateway::signals::CausalProvenanceReport;
use crate::runtime::topology::CORE_TOPOLOGY;
use crate::stigmergy::{
    trail_phase_words_mut, CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, LedgerSlot,
    PhaseAnchorRecord, PointerTrail, ATTRACTOR_SLOT_REPULSION, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
    MAX_ATTRACTOR_PULL_FORCE, TRAIL_FLAG_REPULSION_ACTIVE,
};

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

/// Descriptor header for structure byte size and ABI version validation (8 bytes, 4-byte aligned).
#[repr(C, align(4))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DescriptorHeader {
    /// Size of the enclosing descriptor structure in bytes.
    pub struct_size: u32,
    /// ABI version bitmask matching [`LMNK_ABI_VERSION_HEX`].
    pub abi_version: u32,
}

// SAFETY: `DescriptorHeader` is `#[repr(C, align(4))]` of exact size 8 bytes with zero padding.
unsafe impl Pod for DescriptorHeader {}

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
    assert!(size_of::<DescriptorHeader>() == 8 && align_of::<DescriptorHeader>() == 4);
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

/// Constructs a canonical `DescriptorHeader` for type `T` stamped with [`LMNK_ABI_VERSION_HEX`].
#[inline(always)]
#[must_use]
pub const fn init_descriptor_header<T>() -> DescriptorHeader {
    DescriptorHeader {
        struct_size: size_of::<T>() as u32,
        abi_version: LMNK_ABI_VERSION_HEX,
    }
}

/// Validates a `DescriptorHeader` against expected structure byte size and [`LMNK_ABI_VERSION_HEX`].
///
/// # Errors
/// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `struct_size` or `abi_version` does not match.
#[inline(always)]
pub const fn validate_descriptor_header<T>(header: &DescriptorHeader) -> LmnkResult<()> {
    let valid = (header.struct_size as usize == size_of::<T>())
        & (header.abi_version == LMNK_ABI_VERSION_HEX);
    if valid {
        Ok(())
    } else {
        Err(LMNK_ERR_VERSION_MISMATCH)
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

/// 128-byte cacheline-aligned Plain Old Data (`Pod`) **KV-keyframe Keyframe (I-Frame) Packet** (Section 4.3).
///
/// Packs the 64-byte [`KvKeyframeHeader`] metadata header (`offset 0..64`) and the
/// uncompressed 64-head Q7 phase anchor snapshot (`offset 64..128`) for lightweight video-codec-style
/// `.lmnk` memory & persona streams (portable container distribution ready).
///
/// The 24-byte `header.reserved` field encodes:
/// - `reserved[0..4]`: Little-endian IEEE 802.3 CRC32 checksum of `anchor_phases`.
/// - `reserved[4..8]`: Little-endian 32-bit `lineage_flags` causal mask.
/// - `reserved[8..12]`: Little-endian 32-bit macro Toroidal $k$-Means `cluster_id`.
/// - `reserved[12]`: 3-bit working-lane `slot_id` (`0..=7`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvKeyframePacket {
    /// 64-byte C-ABI KV-keyframe index header (`offset 0..64`, `frame_type == 0`).
    pub header: KvKeyframeHeader,
    /// 64-head Q7 phase anchor snapshot on $SO(2)^{64}$ (`offset 64..128`).
    pub anchor_phases: [PhaseQ7; MAX_ATTENTION_HEADS],
}

// SAFETY: `KvKeyframePacket` is `#[repr(C, align(64))]` of 128 bytes with zero padding traps.
unsafe impl Pod for KvKeyframePacket {}

impl Default for KvKeyframePacket {
    #[inline(always)]
    fn default() -> Self {
        Self {
            header: KvKeyframeHeader {
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
    pub const fn compute_anchor_crc32(anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS]) -> u32 {
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
        lineage_flags: u32,
        cluster_id: u32,
        slot_id: u8,
        anchor_phases: [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> Self {
        let crc = Self::compute_anchor_crc32(&anchor_phases);
        let crc_le = crc.to_le_bytes();
        let prov_le = lineage_flags.to_le_bytes();
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
            header: KvKeyframeHeader {
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

    /// Extracts the 32-bit `lineage_flags` causal mask from `header.reserved[4..8]`.
    #[inline(always)]
    #[must_use]
    pub const fn lineage_flags(&self) -> u32 {
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
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `frame_type != 0` or CRC32 verification fails.
    #[inline(always)]
    pub const fn verify_integrity(&self) -> LmnkResult<()> {
        if self.header.frame_type != 0 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let expected_crc = self.payload_crc32();
        if expected_crc != 0 && Self::compute_anchor_crc32(&self.anchor_phases) != expected_crc {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        Ok(())
    }
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (KV-keyframe) (**Mezzo / Storage Level**).
pub trait KvKeyframeCache: Send + Sync {
    /// Seeks to `token_index` in the KV-keyframe keyframe cache and populates `header_out`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<()>;

    /// Computes Circular Statistics resultant vector length $\bar{R}_c \in [0, 1]$ and von Mises
    /// concentration $\kappa_c$ using $\mathcal{O}(1)$ 256-entry Q7 lookup (`branchless_sincos_q7`)
    /// and division-free (`0 fdiv`) `branchless_inv_pos_f32` (Section 4.2).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `pruned_phases` is empty or its length does not
    /// match `token_weights.len()`.
    #[inline]
    fn aggregate_pruned_kv_circular_stats(
        &self,
        pruned_phases: &AlignedSlice16<'_, PhaseQ7>,
        token_weights: &AlignedSlice32<'_, f32>,
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<(f32, f32)> {
        if pruned_phases.is_empty() || pruned_phases.len() != token_weights.len() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let phases = pruned_phases.as_slice();
        let weights = token_weights.as_slice();
        let n = phases.len();
        let mut sum_c0 = 0.0_f32;
        let mut sum_s0 = 0.0_f32;
        let mut sum_w0 = 0.0_f32;
        let mut sum_c1 = 0.0_f32;
        let mut sum_s1 = 0.0_f32;
        let mut sum_w1 = 0.0_f32;

        let mut idx = 0usize;
        while idx + 2 <= n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;

            let w1 = branchless_relu_f32(weights[idx + 1]);
            let (s1, c1) = branchless_sincos_q7(phases[idx + 1]);
            sum_c1 += w1 * c1;
            sum_s1 += w1 * s1;
            sum_w1 += w1;

            idx += 2;
        }
        if idx < n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;
        }
        let sum_c = sum_c0 + sum_c1;
        let sum_s = sum_s0 + sum_s1;
        let sum_w = sum_w0 + sum_w1;

        let norm = branchless_sqrt_f32(sum_c * sum_c + sum_s * sum_s);
        let r_bar = branchless_flush_subnormal_f32(branchless_select_f32(
            sum_w > DEFAULT_ENTROPY_EPS,
            norm * branchless_inv_pos_f32(sum_w),
            0.0,
        ));
        let r_bar_clamped = branchless_clamp_f32(r_bar, 0.0, 1.0);
        let kappa_c = branchless_von_mises_kappa(r_bar_clamped);
        let theta_bar = branchless_atan2_f32(sum_s, sum_c);

        header_out.composite_vector_length = r_bar_clamped;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = theta_bar;
        Ok((r_bar_clamped, kappa_c))
    }

    /// Computes Multi-Head Virtual Position Shift $\Delta m^{(h)}$ in division-free (`0 fdiv`)
    /// branchless mode (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn compute_virtual_position_shift(
        &self,
        accumulated_phase_rad: f32,
        mean_base_rope_freq: f32,
    ) -> f32 {
        let abs_freq = branchless_abs_f32(mean_base_rope_freq);
        let valid_freq = (abs_freq > 1.0e-9) & branchless_is_finite_f32(mean_base_rope_freq);
        let inv_freq = branchless_copysign_f32(
            branchless_inv_pos_f32(branchless_select_f32(valid_freq, abs_freq, 1.0)),
            mean_base_rope_freq,
        );
        branchless_select_f32(
            valid_freq,
            branchless_flush_subnormal_f32(accumulated_phase_rad * inv_freq),
            0.0,
        )
    }

    /// Evaluates the **Variable Group of Pictures (GOP) Structure** keyframe classification
    /// (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta) and effective GOP length (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn evaluate_variable_gop_trigger(
        &self,
        moving_z_entropy: f32,
        sliding_flux: f32,
        steps_since_keyframe: u32,
        min_gop: u32,
        max_gop: u32,
        entropy_surge_thresh: f32,
        flux_surge_thresh: f32,
    ) -> (u32, u32) {
        let safe_min = branchless_select_u32(min_gop > 0, min_gop, 1);
        let safe_max = branchless_select_u32(max_gop >= safe_min, max_gop, safe_min);
        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(moving_z_entropy),
            moving_z_entropy,
            0.0,
        );
        let clean_flux =
            branchless_select_f32(branchless_is_finite_f32(sliding_flux), sliding_flux, 0.0);

        let surge = (clean_h > entropy_surge_thresh) | (clean_flux > flux_surge_thresh);
        let is_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let frame_type = branchless_select_u32(is_iframe, 0, 1);
        let gop_len = branchless_select_u32(is_iframe, steps_since_keyframe.max(1), safe_max);
        (frame_type, gop_len)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Replay)** phase reconstruction (Section 4.3)
    /// using 8-lane SWAR saturating addition and SWAR `PhaseStepFilter` filtering without forward passes.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_phase_trajectory(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<()> {
        if phase_step_min_q7 > 127 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        *reconstructed_out = *icache_anchor_phases;
        for delta in pcache_motion_deltas {
            branchless_q7_accumulate_64(reconstructed_out, delta);
        }
        branchless_q7_phase_step_filter_64(reconstructed_out, phase_step_min_q7);
        Ok(())
    }

    /// Executes **Seam-Free $S^1$ Geodesic P-Frame Trajectory Replay** (Section 4.3):
    /// accumulates P-Frame motion deltas on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$,
    /// folds onto `[-64, +64]`, and applies 8-lane SWAR step filtering (`PhaseStepFilter`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_geodesic_trajectory(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> LmnkResult<()> {
        if phase_step_min_q7 > 127 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            cur_w[0] = swar_q7_circular_add_x8(cur_w[0], dw[0]);
            cur_w[1] = swar_q7_circular_add_x8(cur_w[1], dw[1]);
            cur_w[2] = swar_q7_circular_add_x8(cur_w[2], dw[2]);
            cur_w[3] = swar_q7_circular_add_x8(cur_w[3], dw[3]);
            cur_w[4] = swar_q7_circular_add_x8(cur_w[4], dw[4]);
            cur_w[5] = swar_q7_circular_add_x8(cur_w[5], dw[5]);
            cur_w[6] = swar_q7_circular_add_x8(cur_w[6], dw[6]);
            cur_w[7] = swar_q7_circular_add_x8(cur_w[7], dw[7]);
        }
        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7);
        store_q7_words_64(reconstructed_out, &filt_w);
        Ok(())
    }

    /// Replays a **Limina P-Frame Motion Chain** ($\boldsymbol{\Delta\theta}_1, \dots, \boldsymbol{\Delta\theta}_T$)
    /// from an **I-Frame Key Anchor** (`icache_anchor_phases`), evaluating the **Joint Probability
    /// Chain Rule Log-Likelihood** $\ln P(M_1, \dots, M_T) = \sum_{t=1}^T \ln P(M_t \mid M_{t-1})$
    /// and inverting any self-contradictory ($H^1(K; \mathcal{F}) \neq 0$) P-Frame step via
    /// Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}_t$).
    ///
    /// Populates `header_out` (`gop_length`, `composite_vector_length = r_bar`,
    /// `von_mises_kappa = kappa_c`, `sparse_scalar_residual = joint_log_prob`) and returns
    /// `(joint_log_prob, r_bar, kappa_c, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127` or `contradiction_threshold` is non-finite.
    #[inline]
    fn replay_liminal_pframe_chain_with_joint_log_prob(
        &self,
        icache_anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[PhaseQ7; MAX_ATTENTION_HEADS]],
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        reconstructed_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        header_out: &mut KvKeyframeHeader,
    ) -> LmnkResult<(f32, f32, f32, u32)> {
        if phase_step_min_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            // Because `next_w = cur_w (+) dw` in Z_256^64 and cos(-d) = cos(d), the transition
            // coherence C(cur_w, next_w) depends purely on `dw` against zero, avoiding a
            // tentative 8-word addition + 8-word subtraction before checking repulsion.
            let coh = coherence_from_diff_words_64(
                dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7],
            );
            let log_p = branchless_fast_transition_log_prob(coh);
            if coh < -safe_thresh {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
                    swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
                    swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
                    swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
                    swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
                    swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
                    swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
                    swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
                ];
                repulsed_count = repulsed_count.wrapping_add(1);
            } else {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], dw[0]),
                    swar_q7_circular_add_x8(cur_w[1], dw[1]),
                    swar_q7_circular_add_x8(cur_w[2], dw[2]),
                    swar_q7_circular_add_x8(cur_w[3], dw[3]),
                    swar_q7_circular_add_x8(cur_w[4], dw[4]),
                    swar_q7_circular_add_x8(cur_w[5], dw[5]),
                    swar_q7_circular_add_x8(cur_w[6], dw[6]),
                    swar_q7_circular_add_x8(cur_w[7], dw[7]),
                ];
            }
            joint_log_prob += log_p;
        }

        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_phase_step_filter_words_64(&folded_w, phase_step_min_q7);
        store_q7_words_64(reconstructed_out, &filt_w);

        let r_bar = circular_r_bar_from_words_64(&filt_w);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        header_out.gop_length = (pcache_motion_deltas.len() as u32).max(1);
        header_out.composite_vector_length = r_bar;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = clean_lp;

        Ok((clean_lp, r_bar, kappa_c, repulsed_count))
    }

    /// Encodes a 128-byte [`KvKeyframePacket`] (**I-Frame Key Anchor**, Section 4.3) from a
    /// 64-head Q7 phase profile (`anchor_phases`), computing Circular Statistics resultant vector
    /// length $\bar{R}_c \in [0, 1]$, von Mises concentration $\kappa_c$, resultant mean phase
    /// angle $\bar{\theta}_c$, and stamping the IEEE 802.3 CRC32 integrity checksum plus causal
    /// provenance metadata (`lineage_flags`, `cluster_id`, `slot_id`).
    #[inline]
    #[must_use]
    fn encode_kv_keyframe_iframe_packet(
        &self,
        token_index: u32,
        gop_length: u32,
        anchor_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        lineage_flags: u32,
        cluster_id: u32,
        slot_id: u8,
    ) -> KvKeyframePacket {
        let words = load_q7_words_64(anchor_phases);
        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&words);
        let theta_bar = branchless_flush_subnormal_f32((mean_q7 as f32) * Q7_PHASE_SCALE_RAD);
        KvKeyframePacket::new(
            token_index,
            gop_length.max(1),
            r_bar,
            kappa_c,
            theta_bar,
            lineage_flags,
            cluster_id,
            slot_id,
            *anchor_phases,
        )
    }

    /// Encodes a 64-byte [`KvMotionDeltaPacket`] (**P-Frame Predictive Phase-Motion Delta**
    /// $\boldsymbol{\Delta\theta}_t$, Section 4.3) between `prev_phases` ($\boldsymbol{\theta}_{t-1}$)
    /// and `curr_phases` ($\boldsymbol{\theta}_t$) using 8-lane SWAR circular difference
    /// (`swar_q7_circular_diff_x8`), $S^1$ geodesic folding (`swar_q7_fold_s1_geodesic_x8`), and
    /// step filtering (`PhaseStepFilter`) (`swar_q7_phase_step_filter_words_64`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `phase_step_min_q7 > 127`.
    #[inline]
    fn encode_kv_keyframe_pframe_delta_packet(
        &self,
        prev_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        curr_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
        phase_step_min_q7: u8,
    ) -> LmnkResult<KvMotionDeltaPacket> {
        encode_pframe_motion_delta_64(prev_phases, curr_phases, phase_step_min_q7)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Persona & Memory Replay)** over a
    /// lightweight KV-keyframe Variable GOP stream (`keyframe` I-Frame + `pframes[..seek_pframe_steps]`
    /// P-Frame deltas), reconstructing the historical 64-head persona & episodic memory state
    /// directly onto `trail_out` without running any LLM forward pass (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if `seek_pframe_steps > pframes.len()`,
    /// `phase_step_min_q7 > 127`, or `contradiction_threshold` is non-finite.
    #[inline]
    fn seek_and_reconstruct_persona_from_gop(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        seek_pframe_steps: usize,
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(f32, f32, f32, f32, u32)> {
        keyframe.verify_integrity()?;
        let (filt_w, clean_lp, repulsed_count) = seek_pframe_chain_words_64(
            &keyframe.anchor_phases,
            pframes,
            seek_pframe_steps,
            phase_step_min_q7,
            contradiction_threshold,
        )?;
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(TaggedPtr(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe
            .header
            .token_index
            .wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = TaggedPtr::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.lineage_flags());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }

    /// Serializes a lightweight KV-keyframe Variable GOP stream (`1` 128-byte [`KvKeyframePacket`] +
    /// `pframes.len()` 64-byte [`KvMotionDeltaPacket`]s) into `dst_bytes` (`128 + N * 64` bytes),
    /// returning the total byte length written.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`LMNK_ERR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < 128 + pframes.len() * 64`.
    #[inline]
    fn pack_kv_gop_stream_bytes(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        keyframe.verify_integrity()?;
        let p_bytes = pframes
            .len()
            .checked_mul(core::mem::size_of::<KvMotionDeltaPacket>())
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        let total_bytes = core::mem::size_of::<KvKeyframePacket>()
            .checked_add(p_bytes)
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if dst_bytes.len() < total_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        keyframe.write_unaligned_to(dst_bytes, 0)?;
        KvMotionDeltaPacket::write_unaligned_slice_to(
            pframes,
            dst_bytes,
            core::mem::size_of::<KvKeyframePacket>(),
        )?;
        Ok(total_bytes)
    }

    /// Unpacks a serialized KV-keyframe Variable GOP byte stream (`stream_bytes`) in $\mathcal{O}(1)$
    /// stack space ($C_{\mathrm{alloc}} = 0$) and executes **Zero-Forward Latency Seek** up to
    /// `seek_pframe_steps`, reconstructing the historical persona and episodic memory state onto
    /// `trail_out` (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `stream_bytes` is truncated,
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`] if the I-Frame header or CRC32 is invalid, or
    /// [`LMNK_ERR_INVALID_ARGUMENT`] if parameters are out of range.
    #[inline]
    fn unpack_and_seek_kv_gop_stream_bytes(
        &self,
        stream_bytes: &[u8],
        seek_pframe_steps: usize,
        phase_step_min_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(f32, f32, f32, f32, u32)> {
        if phase_step_min_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let iframe_size = core::mem::size_of::<KvKeyframePacket>();
        let pframe_size = core::mem::size_of::<KvMotionDeltaPacket>();
        let needed_bytes = seek_pframe_steps
            .checked_mul(pframe_size)
            .and_then(|b| b.checked_add(iframe_size))
            .ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
        if stream_bytes.len() < needed_bytes {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let keyframe = KvKeyframePacket::read_unaligned_from(stream_bytes, 0)?;
        keyframe.verify_integrity()?;

        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(&keyframe.anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        let mut step = 0usize;
        while step < seek_pframe_steps {
            let off = iframe_size + step * pframe_size;
            let pframe = KvMotionDeltaPacket::read_unaligned_from(stream_bytes, off)?;
            let (log_p, repulsed) =
                apply_pframe_motion_step_words_64(&mut cur_w, &pframe, safe_thresh);
            joint_log_prob += log_p;
            if repulsed {
                repulsed_count = repulsed_count.wrapping_add(1);
            }
            step += 1;
        }

        let filt_w = finalize_pframe_chain_words_64(&cur_w, phase_step_min_q7);
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(TaggedPtr(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe
            .header
            .token_index
            .wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = TaggedPtr::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.lineage_flags());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }
}

// ===================================================================
// Compact 64-Byte Phase Synchronization Packet Helpers
// ===================================================================

/// Dispatches a compact 64-byte phase/repulsion synchronization packet in zero-allocation mode.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1` (Network Plugin Mode).
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
#[inline]
pub fn dispatch_phase_packet(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
) -> LmnkResult<()> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_payload.len() < 64 {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    Ok(())
}

/// Encodes a 64-head Q7 phase/repulsion vector into a 64-byte cacheline-aligned phase
/// packet (`packet_out`) using bulk `Pod` copy and returns its branchless IEEE 802.3 CRC32 integrity tag.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_out.len() < 64` or `source_slot_id >= 8`.
#[inline]
pub fn encode_phase_packet(
    ledger_slot: &LedgerSlot,
    source_slot_id: u8,
    action_mode_flags: u8,
    phase_deltas: &[PhaseQ7; MAX_ATTENTION_HEADS],
    packet_out: &mut AlignedSliceMut64<'_, u8>,
) -> LmnkResult<u32> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_out.len() < MAX_ATTENTION_HEADS || (source_slot_id as usize) >= MAX_ATTRACTOR_DOMAINS
    {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let phase_w = load_q7_words_64(phase_deltas);
    let dst = &mut packet_out.as_mut_slice()[..MAX_ATTENTION_HEADS];
    let mut cell = 0usize;
    while cell < 8 {
        let base = cell << 3;
        dst[base..base + 8].copy_from_slice(&phase_w[cell].to_le_bytes());
        cell += 1;
    }
    let tag_byte = pack_action_slot_tag(source_slot_id, action_mode_flags);
    let crc = crc32_block64_zero_seed_words(&phase_w) ^ u32::from(tag_byte);
    Ok(crc)
}

/// Verifies and decodes a 64-byte cacheline-aligned phase packet (`packet_payload`) into
/// 8 register-resident `u64` SWAR words and the recovered 8-bit tag byte.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
/// checksum do not match `expected_crc32`.
#[inline(always)]
pub fn decode_and_verify_phase_packet_words(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
) -> LmnkResult<([u64; 8], u8)> {
    if ledger_slot.mode != 1 {
        return Err(LMNK_ERR_INVALID_STATE);
    }
    if packet_payload.len() < MAX_ATTENTION_HEADS {
        return Err(LMNK_ERR_INVALID_ARGUMENT);
    }
    let src = &packet_payload.as_slice()[..MAX_ATTENTION_HEADS];
    let payload_words = [
        u64::from_le_bytes([
            src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
        ]),
        u64::from_le_bytes([
            src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15],
        ]),
        u64::from_le_bytes([
            src[16], src[17], src[18], src[19], src[20], src[21], src[22], src[23],
        ]),
        u64::from_le_bytes([
            src[24], src[25], src[26], src[27], src[28], src[29], src[30], src[31],
        ]),
        u64::from_le_bytes([
            src[32], src[33], src[34], src[35], src[36], src[37], src[38], src[39],
        ]),
        u64::from_le_bytes([
            src[40], src[41], src[42], src[43], src[44], src[45], src[46], src[47],
        ]),
        u64::from_le_bytes([
            src[48], src[49], src[50], src[51], src[52], src[53], src[54], src[55],
        ]),
        u64::from_le_bytes([
            src[56], src[57], src[58], src[59], src[60], src[61], src[62], src[63],
        ]),
    ];
    let raw_crc = crc32_block64_zero_seed_words(&payload_words);
    let diff = raw_crc ^ expected_crc32;
    if (diff & 0xFFFF_FF00) != 0 {
        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
    }
    Ok((payload_words, (diff & 0xFF) as u8))
}

/// Verifies and decodes a 64-byte cacheline-aligned LiminiKa Phase Packet (`packet_payload`) into
/// a 64-head Q7 phase/repulsion vector (`phases_out`), returning the recovered 8-bit tag byte.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_STATE`] if `ledger_slot.mode != 1`.
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
/// checksum do not match `expected_crc32`.
#[inline]
pub fn decode_and_verify_phase_packet(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
    phases_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
) -> LmnkResult<u8> {
    match decode_and_verify_phase_packet_words(ledger_slot, packet_payload, expected_crc32) {
        Ok((payload_words, tag_byte)) => {
            store_q7_words_64(phases_out, &payload_words);
            Ok(tag_byte)
        }
        Err(err) => {
            if err == LMNK_ERR_CONTAINER_CORRUPTED {
                *phases_out = [0i8; MAX_ATTENTION_HEADS];
            }
            Err(err)
        }
    }
}

/// Verifies a 64-byte LiminiKa Phase Packet and applies its decoded 64-head phase/repulsion vector
/// directly to `trail` (and `bloom_filter` if repulsive).
///
/// # Errors
/// Propagates any error returned by [`decode_and_verify_phase_packet`].
#[inline]
pub fn decode_and_apply_phase_packet_to_trail(
    ledger_slot: &LedgerSlot,
    packet_payload: &AlignedSlice64<'_, u8>,
    expected_crc32: u32,
    inhibition_strength: f32,
    bloom_filter: Option<&mut ContrastivePhaseBloomFilter>,
    trail: &mut PointerTrail,
) -> LmnkResult<(u8, u8, bool)> {
    let (payload_words, tag_byte) =
        decode_and_verify_phase_packet_words(ledger_slot, packet_payload, expected_crc32)?;
    let slot_id = unpack_attractor_slot_from_tag(tag_byte);
    let mode_flags = unpack_action_mode_from_tag(tag_byte);
    let is_repulsion =
        (slot_id == ATTRACTOR_SLOT_REPULSION) | ((mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);

    if is_repulsion {
        if let Some(bf) = bloom_filter {
            bf.insert_repulsive_pattern_words_weighted(&payload_words, 1);
        }
        let cur_pull = trail.attractor_pull_force;
        let damped = branchless_q7_inhibit_words_64(
            trail_phase_words_mut(trail),
            cur_pull,
            &payload_words,
            inhibition_strength,
        );
        trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(damped),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
    } else {
        let tw = trail_phase_words_mut(trail);
        tw[0] = swar_q7_add_sat_x8(tw[0], payload_words[0]);
        tw[1] = swar_q7_add_sat_x8(tw[1], payload_words[1]);
        tw[2] = swar_q7_add_sat_x8(tw[2], payload_words[2]);
        tw[3] = swar_q7_add_sat_x8(tw[3], payload_words[3]);
        tw[4] = swar_q7_add_sat_x8(tw[4], payload_words[4]);
        tw[5] = swar_q7_add_sat_x8(tw[5], payload_words[5]);
        tw[6] = swar_q7_add_sat_x8(tw[6], payload_words[6]);
        tw[7] = swar_q7_add_sat_x8(tw[7], payload_words[7]);
    }
    Ok((slot_id, mode_flags, is_repulsion))
}

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
    /// Section kind ID ([`LmnkSectionKind`] or custom `u16`).
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
    /// 2. Magic (`"LMNK"`), ABI version (`0x0000_0200`), header geometry (`header_size == 64`,
    ///    `section_table_offset == 64`, `section_count <= 32`).
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
        if header.version != LMNK_ABI_VERSION_HEX {
            return Err(LMNK_ERR_VERSION_MISMATCH);
        }
        if header.header_size != LMNK_CONTAINER_HEADER_BYTES as u16
            || header.section_table_offset != LMNK_CONTAINER_HEADER_BYTES as u32
            || header.section_count > LMNK_MAX_CONTAINER_SECTIONS
            || header.total_size != bytes.len() as u64
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
        if bytes.len() < table_end {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let mut prev_end = align_up_64_checked(table_end).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
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

            if entry.length == 0 {
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
            if (sec_off & 63) != 0 || sec_off < prev_end {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
            let sec_end = sec_off
                .checked_add(sec_len)
                .ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
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
            prev_end = align_up_64_checked(sec_end).ok_or(LMNK_ERR_CONTAINER_CORRUPTED)?;
            i += 1;
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
            if let Some(&entry) = self.sections.get(idx) {
                if entry.kind == kind {
                    return Some(entry);
                }
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
        num_crystals: usize,
        num_trails: usize,
        has_bloom: bool,
        has_ledger: bool,
        num_reports: usize,
    ) -> LmnkResult<usize> {
        let sec_count = (num_crystals > 0) as usize
            + (num_trails > 0) as usize
            + (has_bloom as usize)
            + (has_ledger as usize)
            + (num_reports > 0) as usize;
        let table_end = LMNK_CONTAINER_HEADER_BYTES + sec_count * LMNK_SECTION_ENTRY_BYTES;
        let Some(mut cursor) = align_up_64_checked(table_end) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };

        if num_crystals > 0 {
            let Some(bytes) = num_crystals.checked_mul(size_of::<PhaseAnchorRecord>()) else {
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
    #[allow(clippy::too_many_arguments)]
    pub fn write_standard_sections_to_bytes(
        container_flags: u32,
        crystals: &[PhaseAnchorRecord],
        trails: &[PointerTrail],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        ledger_opt: Option<&LedgerSlot>,
        provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        let total_len = Self::required_standard_byte_len(
            crystals.len(),
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

        let sec_count = usize::from(!crystals.is_empty())
            + usize::from(!trails.is_empty())
            + usize::from(bloom_opt.is_some())
            + usize::from(ledger_opt.is_some())
            + usize::from(!provenance_reports.is_empty());
        let table_end = LMNK_CONTAINER_HEADER_BYTES + sec_count * LMNK_SECTION_ENTRY_BYTES;
        let mut cursor = align_up_64_checked(table_end).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let mut sec_idx = 0usize;

        if !crystals.is_empty() {
            let off = cursor;
            let len = crystals.len() * size_of::<PhaseAnchorRecord>();
            let end = off + len;
            let payload_slice = buf.get_mut(off..end).ok_or(LMNK_ERR_BUFFER_TOO_SMALL)?;
            let _ = PhaseAnchorRecord::write_records_to_container_bytes(crystals, payload_slice)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{AlignedBuffer128, AlignedBuffer64};
    use crate::liminal::{LMNK_ERR_ALIGNMENT, MAX_ATTENTION_HEADS};
    use crate::runtime::gateway::CAUSAL_AXIS_SUBJECTIVE_PERSONA;
    use crate::stigmergy::ATTRACTOR_SLOT_PERSONA;

    fn restamp_container_header_crc(buf: &mut [u8]) {
        let crc = compute_crc32_ieee(&buf[0..60]);
        buf[60..64].copy_from_slice(&crc.to_le_bytes());
    }

    #[test]
    fn test_container_view_hardened_parsing_alignment_and_reserved_sections() {
        let anchors = [
            PhaseAnchorRecord::new_empty(11, ATTRACTOR_SLOT_PERSONA, 0x1000),
            PhaseAnchorRecord::new_empty(22, ATTRACTOR_SLOT_PERSONA, 0x2000),
        ];
        let trails = [PointerTrail::default()];
        let bloom = ContrastivePhaseBloomFilter::new();
        let ledger = LedgerSlot::default();
        let reports = [CausalProvenanceReport::default()];

        let req_len = LmnkContainerView::required_standard_byte_len(
            anchors.len(),
            trails.len(),
            true,
            true,
            reports.len(),
        )
        .unwrap();
        let mut aligned = AlignedBuffer128::<u8, 2048>::new([0u8; 2048]);
        let written = LmnkContainerView::write_standard_sections_to_bytes(
            LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            &anchors,
            &trails,
            Some(&bloom),
            Some(&ledger),
            &reports,
            1_774_900_000,
            &mut aligned.data[..req_len],
        )
        .unwrap();
        assert_eq!(written, req_len);

        // Aligned zero-copy parse succeeds and yields typed slices.
        let valid_view = LmnkContainerView::parse(&aligned.data[..written]).unwrap();
        assert_eq!(
            valid_view.header().topology_crc32,
            CORE_TOPOLOGY.topology_crc32
        );
        assert_eq!(valid_view.sections().len(), 5);
        assert_eq!(valid_view.phase_anchors_zero_copy().unwrap().len(), 2);
        assert_eq!(valid_view.pointer_trails_zero_copy().unwrap().len(), 1);
        assert!(valid_view.bloom_filter_zero_copy().unwrap().is_some());
        assert!(valid_view.ledger_slot_zero_copy().unwrap().is_some());
        assert_eq!(valid_view.causal_provenance_zero_copy().unwrap().len(), 1);

        // Intentionally 1-byte-misaligned slice: parse succeeds, zero-copy typed slice returns LMNK_ERR_ALIGNMENT,
        // while section_bytes() and read_phase_anchor() succeed cleanly!
        let mut unaligned_backing = alloc::vec![0u8; written + 2];
        unaligned_backing[1..1 + written].copy_from_slice(&aligned.data[..written]);
        let misaligned = &unaligned_backing[1..1 + written];
        let mis_view = LmnkContainerView::parse(misaligned).unwrap();
        assert_eq!(mis_view.phase_anchors_zero_copy(), Err(LMNK_ERR_ALIGNMENT));
        assert_eq!(mis_view.pointer_trails_zero_copy(), Err(LMNK_ERR_ALIGNMENT));
        assert_eq!(mis_view.bloom_filter_zero_copy(), Err(LMNK_ERR_ALIGNMENT));
        assert_eq!(mis_view.ledger_slot_zero_copy(), Err(LMNK_ERR_ALIGNMENT));
        assert_eq!(
            mis_view
                .section_bytes(LMNK_SECTION_PHASE_ANCHORS)
                .unwrap()
                .len(),
            256
        );
        assert_eq!(mis_view.read_phase_anchor(0).unwrap().anchor_id, 11);
        assert_eq!(mis_view.read_phase_anchor(1).unwrap().anchor_id, 22);

        // Reserved ContentHash (0xFF01) & Signature (0xFF02) sections accepted cleanly.
        let digest = [0xABu8; 32];
        let sig = [0xCDu8; 64];
        let sec_inputs = [
            LmnkSectionInput {
                kind: LMNK_SECTION_CONTENT_HASH,
                flags: 0,
                payload: &digest,
            },
            LmnkSectionInput {
                kind: LMNK_SECTION_SIGNATURE,
                flags: 0,
                payload: &sig,
            },
        ];
        let sec_len = required_container_byte_len(&sec_inputs).unwrap();
        let mut sec_buf = alloc::vec![0u8; sec_len];
        write_container_sections_to_bytes(
            LMNK_CONTAINER_FLAG_PORTABLE_PACKAGE,
            1_774_900_000,
            &sec_inputs,
            &mut sec_buf,
        )
        .unwrap();
        let sec_view = LmnkContainerView::parse(&sec_buf).unwrap();
        assert_eq!(
            sec_view.section_bytes(LMNK_SECTION_CONTENT_HASH),
            Some(&digest[..])
        );
        assert_eq!(
            sec_view.section_bytes(LMNK_SECTION_SIGNATURE),
            Some(&sig[..])
        );

        // Fault injection suite (zero panics):
        // 1. Truncated buffer
        assert_eq!(
            LmnkContainerView::parse(&aligned.data[..63]),
            Err(LMNK_ERR_BUFFER_TOO_SMALL)
        );
        // 2. total_size mismatch
        let mut bad = aligned.data[..written].to_vec();
        bad[16..24].copy_from_slice(&((written as u64) + 64).to_le_bytes());
        restamp_container_header_crc(&mut bad);
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 3. Bad magic
        let mut bad = aligned.data[..written].to_vec();
        bad[0] ^= 0xFF;
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 4. Bad version
        let mut bad = aligned.data[..written].to_vec();
        bad[4..8].copy_from_slice(&0x0000_0103u32.to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_VERSION_MISMATCH)
        );
        // 5. Corrupt header CRC32
        let mut bad = aligned.data[..written].to_vec();
        bad[60] ^= 0xFF;
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 6. Mismatched topology_crc32 (with valid header_crc32) -> LMNK_ERR_TOPOLOGY_MISMATCH
        let mut bad = aligned.data[..written].to_vec();
        bad[8..12].copy_from_slice(&(CORE_TOPOLOGY.topology_crc32 ^ 0x1234_5678).to_le_bytes());
        restamp_container_header_crc(&mut bad);
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_TOPOLOGY_MISMATCH)
        );
        // 7. section_count > 32
        let mut bad = aligned.data[..written].to_vec();
        bad[36..38].copy_from_slice(&33u16.to_le_bytes());
        restamp_container_header_crc(&mut bad);
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 8. length == 0
        let mut bad = aligned.data[..written].to_vec();
        bad[64 + 8..64 + 12].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 9. Duplicate section kind
        let mut bad = aligned.data[..written].to_vec();
        bad[64 + 16..64 + 18].copy_from_slice(&LMNK_SECTION_PHASE_ANCHORS.to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 10. Unaligned section offset (offset % 64 != 0)
        let mut bad = aligned.data[..written].to_vec();
        let orig_off = u32::from_le_bytes([bad[64 + 4], bad[64 + 5], bad[64 + 6], bad[64 + 7]]);
        bad[64 + 4..64 + 8].copy_from_slice(&(orig_off + 8).to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 11. Overlapping / out-of-order sections (second section offset == first section offset)
        let mut bad = aligned.data[..written].to_vec();
        bad[64 + 16 + 4..64 + 16 + 8].copy_from_slice(&orig_off.to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 12. Integer-overflowing offset + length
        let mut bad = aligned.data[..written].to_vec();
        bad[64 + 8..64 + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
        // 13. Corrupt payload CRC32
        let mut bad = aligned.data[..written].to_vec();
        bad[orig_off as usize] ^= 0x01;
        assert_eq!(
            LmnkContainerView::parse(&bad),
            Err(LMNK_ERR_CONTAINER_CORRUPTED)
        );
    }

    #[test]
    fn test_mapped_storage_engine_phase_packet_and_gop_seek() {
        let engine = MappedStorageEngine::new(4096, 1);
        assert_eq!(engine.validate(), Ok(()));

        let ledger_slot = LedgerSlot {
            mode: 1,
            ..Default::default()
        };
        let mut packet_buf = AlignedBuffer64::new([0u8; 64]);
        let phases = [12i8; MAX_ATTENTION_HEADS];
        let crc = encode_phase_packet(
            &ledger_slot,
            2,
            0x09,
            &phases,
            &mut packet_buf.as_aligned_slice_mut(),
        )
        .unwrap();
        let mut trail = PointerTrail::default();
        let (slot, mode, is_repulsion) = decode_and_apply_phase_packet_to_trail(
            &ledger_slot,
            &packet_buf.as_aligned_slice(),
            crc,
            0.5,
            None,
            &mut trail,
        )
        .unwrap();
        assert_eq!(slot, 2);
        assert_eq!(mode, 0x09);
        assert!(!is_repulsion);
        assert_eq!(trail.accumulated_phase_delta[0], 12);

        let rep_crc = encode_phase_packet(
            &ledger_slot,
            3,
            0x05,
            &phases,
            &mut packet_buf.as_aligned_slice_mut(),
        )
        .unwrap();
        let mut rep_trail = PointerTrail::default();
        let (rep_slot, rep_mode, rep_is_repulsion) = decode_and_apply_phase_packet_to_trail(
            &ledger_slot,
            &packet_buf.as_aligned_slice(),
            rep_crc,
            0.5,
            None,
            &mut rep_trail,
        )
        .unwrap();
        assert_eq!(rep_slot, 3);
        assert_eq!(rep_mode, 0x05);
        assert!(rep_is_repulsion);
        assert_eq!(rep_trail.accumulated_phase_delta[0], 0);
        assert_ne!(rep_trail.flags, 0);

        // Verify KV-keyframe Video-Codec Style I-Frame / P-Frame GOP Stream & Zero-Forward Latency Seek
        let anchor_t0 = [16i8; MAX_ATTENTION_HEADS];
        let state_t1 = [20i8; MAX_ATTENTION_HEADS];
        let state_t2 = [26i8; MAX_ATTENTION_HEADS];

        let iframe = engine.encode_kv_keyframe_iframe_packet(
            100,
            3,
            &anchor_t0,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            7,
            ATTRACTOR_SLOT_PERSONA,
        );
        assert_eq!(iframe.verify_integrity(), Ok(()));
        assert_eq!(iframe.cluster_id(), 7);
        assert_eq!(iframe.slot_id(), ATTRACTOR_SLOT_PERSONA);

        let pframe1 = engine
            .encode_kv_keyframe_pframe_delta_packet(&anchor_t0, &state_t1, 1)
            .unwrap();
        let pframe2 = engine
            .encode_kv_keyframe_pframe_delta_packet(&state_t1, &state_t2, 1)
            .unwrap();
        let gop_pframes = [pframe1, pframe2];

        let mut stream_bytes = [0u8; 256];
        let written = engine
            .pack_kv_gop_stream_bytes(&iframe, &gop_pframes, &mut stream_bytes)
            .unwrap();
        assert_eq!(written, 256);

        let mut seek_trail = PointerTrail::default();
        let (lp, r_bar, kappa, v_shift, rep_cnt) = engine
            .unpack_and_seek_kv_gop_stream_bytes(
                &stream_bytes[..written],
                2,
                1,
                0.25,
                0.1,
                &mut seek_trail,
            )
            .unwrap();
        assert_eq!(rep_cnt, 0);
        assert_eq!(seek_trail.accumulated_phase_delta[0], 26);
        assert_eq!(seek_trail.cluster_id, 7);
        assert_eq!(
            (seek_trail.user_data as u32) & CAUSAL_AXIS_SUBJECTIVE_PERSONA,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA
        );
        assert_eq!(TaggedPtr(seek_trail.current_ptr).unpack_addr(), 102);
        assert!(lp > -0.05 && (r_bar - 1.0).abs() < 1.0e-4 && kappa > 10.0 && v_shift > 0.0);
    }
}
