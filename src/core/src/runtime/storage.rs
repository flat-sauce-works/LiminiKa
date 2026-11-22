// name: src/core/src/runtime/storage.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-Copy In-Memory Mapped Storage (`MappedStorageEngine` 64B `Pod`,
//! [`LmnkFourSectionView`] 4-section `.lmnk` container zero-copy persistence & evolution,
//! [`KvKeyframeCache`], [`MappedStorage`], phase packet codecs, and [`KvKeyframePacket`]).

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
    crc32_slice16_indep_hi1_u32, crc32_slice16_indep_lo1_u32, crc32_slice16_indep_u64x2,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l2, crc32_step_l3, crc32_step_l32,
    crc32_step_l4, crc32_step_l8, encode_pframe_motion_delta_64, finalize_pframe_chain_words_64,
    is_aligned_to, load_q7_words_64, pack_action_slot_tag, seek_pframe_chain_words_64,
    store_q7_words_64, swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_fold_s1_geodesic_x8,
    swar_q7_neg_sat_x8, swar_q7_phase_step_filter_words_64, unpack_action_mode_from_tag,
    unpack_attractor_slot_from_tag, update_aligned64_blocks_crc32_raw, AlignedSlice,
    AlignedSlice16, AlignedSlice32, AlignedSlice64, AlignedSliceMut64, Pod, TaggedPtr,
    ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, LmnkResult, PhaseQ7, RuntimeConfig,
    DEFAULT_ENTROPY_EPS, LMNK_ABI_VERSION_HEX, LMNK_ERR_ALIGNMENT, LMNK_ERR_BUFFER_TOO_SMALL,
    LMNK_ERR_CONTAINER_CORRUPTED, LMNK_ERR_INVALID_ARGUMENT, LMNK_ERR_INVALID_STATE,
    LMNK_ERR_VERSION_MISMATCH, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use crate::runtime::gateway::signals::CausalProvenanceReport;
use crate::stigmergy::anchor::{crc32_anchor_stamp_contrib_128b, crc32_fold_stamped_anchor_128b};
use crate::stigmergy::{
    trail_phase_words_mut, CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, LedgerSlot,
    PhaseAnchorRecord, PointerTrail, ATTRACTOR_SLOT_REPULSION, LEDGER_FLAG_COGNITIVE_TRAJECTORY,
    MAX_ATTRACTOR_PULL_FORCE, TRAIL_FLAG_REPULSION_ACTIVE,
};

/// Binary magic constant for `.lmnk` snapshot headers (`0x4B4E4D4C` = ASCII `"LMNK"`).
pub const LMNK_SNAPSHOT_MAGIC_U32: u32 = 0x4B4E_4D4C;

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

/// Internal 64-byte preamble (`#[repr(C, align(32))]`) stored in Section 0b (`offset 128..192`)
/// of 4-section `.lmnk` container views. Kept crate-private outside the public C-ABI surface.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MappedStoragePreamble {
    pub(crate) mapped_address: u64,
    pub(crate) file_size_bytes: u64,
    pub(crate) dma_buffer_handle: u64,
    pub(crate) flags: u32,
    pub(crate) fd_handle: i32,
    pub(crate) reserved: [u8; 32],
}

impl Default for MappedStoragePreamble {
    #[inline(always)]
    fn default() -> Self {
        Self {
            mapped_address: 0,
            file_size_bytes: 0,
            dma_buffer_handle: 0,
            flags: 0,
            fd_handle: -1,
            reserved: [0; 32],
        }
    }
}

// SAFETY: `MappedStoragePreamble` is `#[repr(C, align(32))]` of exact size 64 bytes with zero padding.
unsafe impl Pod for MappedStoragePreamble {}

/// Unified `.lmnk` binary snapshot container header structure (128 bytes, 64-byte aligned).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LmnkSnapshotHeader {
    /// Format identifier magic constant (`0x4B4E4D4C` = ASCII `"LMNK"`).
    pub magic: u32,
    /// Snapshot ABI version matching [`LMNK_ABI_VERSION_HEX`].
    pub version: u32,
    /// Total binary size of container payload in bytes.
    pub total_size: u64,
    /// Byte offset to substrate section.
    pub substrate_offset: u64,
    /// Byte offset to anchor section.
    pub anchor_offset: u64,
    /// Byte offset to phase-steering section.
    pub phase_steering_offset: u64,
    /// Byte offset to Rank-1 residual adapter section.
    pub rank1_offset: u64,
    /// Byte offset to entropy controller section.
    pub entropy_offset: u64,
    /// IEEE 802.3 CRC32 integrity checksum over snapshot payload.
    pub checksum_crc32: u32,
    /// Byte offset to fast-path ledger slot section.
    pub ledger_slot_offset: u32,
    /// Epoch timestamp of snapshot serialization (seconds).
    pub timestamp_epoch_sec: u64,
    /// Explicit padding to enforce exact 128-byte size and 64-byte cacheline alignment (56 bytes).
    pub reserved_padding: [u8; 56],
}

impl Default for LmnkSnapshotHeader {
    #[inline(always)]
    fn default() -> Self {
        Self {
            magic: 0,
            version: LMNK_ABI_VERSION_HEX,
            total_size: 0,
            substrate_offset: 0,
            anchor_offset: 0,
            phase_steering_offset: 0,
            rank1_offset: 0,
            entropy_offset: 0,
            checksum_crc32: 0,
            ledger_slot_offset: 0,
            timestamp_epoch_sec: 0,
            reserved_padding: [0; 56],
        }
    }
}

// SAFETY: `LmnkSnapshotHeader` is `#[repr(C, align(64))]` of exact size 128 bytes with zero padding.
unsafe impl Pod for LmnkSnapshotHeader {}

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
    assert!(size_of::<MappedStoragePreamble>() == 64 && align_of::<MappedStoragePreamble>() == 32);
    assert!(size_of::<LmnkSnapshotHeader>() == 128 && align_of::<LmnkSnapshotHeader>() == 64);
    assert!(offset_of!(LmnkSnapshotHeader, magic) == 0);
    assert!(offset_of!(LmnkSnapshotHeader, version) == 4);
    assert!(offset_of!(LmnkSnapshotHeader, total_size) == 8);
    assert!(offset_of!(LmnkSnapshotHeader, substrate_offset) == 16);
    assert!(offset_of!(LmnkSnapshotHeader, anchor_offset) == 24);
    assert!(offset_of!(LmnkSnapshotHeader, phase_steering_offset) == 32);
    assert!(offset_of!(LmnkSnapshotHeader, rank1_offset) == 40);
    assert!(offset_of!(LmnkSnapshotHeader, entropy_offset) == 48);
    assert!(offset_of!(LmnkSnapshotHeader, checksum_crc32) == 56);
    assert!(offset_of!(LmnkSnapshotHeader, ledger_slot_offset) == 60);
    assert!(offset_of!(LmnkSnapshotHeader, timestamp_epoch_sec) == 64);
    assert!(offset_of!(LmnkSnapshotHeader, reserved_padding) == 72);
    assert!(size_of::<KvKeyframeHeader>() == 64 && align_of::<KvKeyframeHeader>() == 32);
};

/// Precomputed 64-byte GF(2) shift $L^4(\text{0xFFFF\_FFFF})$ of the initial IEEE 802.3 CRC32 seed.
const CRC32_L4_INITIAL_SEED: u32 = crc32_step_l4(0xFFFF_FFFFu32);

/// Precomputed 64-byte IEEE 802.3 CRC32 accumulator state over the invariant fields of a default
/// 4-section `.lmnk` `MappedStoragePreamble` (`mapped_address = 0`, `file_size_bytes = 0`,
/// `dma_buffer_handle = 0`, `flags = 4`, `fd_handle = -1`, `reserved = [0; 32]`).
const CRC32_MAPPED_STORAGE_DESC_BASE_SEED: u32 =
    CRC32_L4_INITIAL_SEED ^ crc32_step_l2(crc32_slice16_indep_u64x2(0, 0xFFFF_FFFF_0000_0004u64));

/// Fixed byte length (`192` bytes = `128B` `LmnkSnapshotHeader` + `64B` `MappedStoragePreamble`) of Section 0.
pub const LMNK_FOUR_SECTION_HEADER_BYTES: usize = 192;

/// Byte offset (`128`) within the 192-byte Section 0 preamble where `MappedStoragePreamble` is stored.
pub const LMNK_STORAGE_PREAMBLE_OFFSET: usize = 128;

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

/// Constructs a canonical `LmnkSnapshotHeader` initialized with magic `"LMNK"` (`0x4B4E_4D4C`),
/// [`LMNK_ABI_VERSION_HEX`], and `total_size`.
#[inline(always)]
#[must_use]
pub const fn init_snapshot_header(total_size: u64, timestamp_epoch_sec: u64) -> LmnkSnapshotHeader {
    let hdr_size = size_of::<LmnkSnapshotHeader>() as u64;
    LmnkSnapshotHeader {
        magic: LMNK_SNAPSHOT_MAGIC_U32,
        version: LMNK_ABI_VERSION_HEX,
        total_size,
        substrate_offset: hdr_size,
        anchor_offset: hdr_size,
        phase_steering_offset: hdr_size,
        rank1_offset: hdr_size,
        entropy_offset: hdr_size,
        checksum_crc32: 0,
        ledger_slot_offset: hdr_size as u32,
        timestamp_epoch_sec,
        reserved_padding: [0; 56],
    }
}

/// Validates a `LmnkSnapshotHeader` against the `.lmnk` container magic (`0x4B4E4D4C`),
/// [`LMNK_ABI_VERSION_HEX`], and payload buffer bounds.
///
/// # Errors
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `magic` is invalid, `total_size` is smaller than
/// `size_of::<LmnkSnapshotHeader>()`, `available_bytes < total_size`, or any track offset exceeds `total_size`.
/// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `version != LMNK_ABI_VERSION_HEX`.
#[inline]
pub const fn validate_snapshot_header(
    header: &LmnkSnapshotHeader,
    available_bytes: usize,
) -> LmnkResult<()> {
    if header.magic != LMNK_SNAPSHOT_MAGIC_U32 {
        return Err(LMNK_ERR_CONTAINER_CORRUPTED);
    }
    if header.version != LMNK_ABI_VERSION_HEX {
        return Err(LMNK_ERR_VERSION_MISMATCH);
    }
    let min_hdr = size_of::<LmnkSnapshotHeader>() as u64;
    let total = header.total_size;
    let ledger_off = header.ledger_slot_offset as u64;
    let offsets_in_bounds = (total >= min_hdr)
        & ((available_bytes as u64) >= total)
        & (header.substrate_offset >= min_hdr)
        & (header.substrate_offset <= total)
        & (header.anchor_offset >= min_hdr)
        & (header.anchor_offset <= total)
        & (header.phase_steering_offset >= min_hdr)
        & (header.phase_steering_offset <= total)
        & (header.rank1_offset >= min_hdr)
        & (header.rank1_offset <= total)
        & (header.entropy_offset >= min_hdr)
        & (header.entropy_offset <= total)
        & (ledger_off >= min_hdr)
        & (ledger_off <= total);
    if offsets_in_bounds {
        Ok(())
    } else {
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    }
}

/// Validates a `.lmnk` snapshot header, computes the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`), and stamps
/// `header.checksum_crc32` in-place.
///
/// # Errors
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if header offsets or payload bounds are invalid.
/// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `header.version != LMNK_ABI_VERSION_HEX`.
#[inline]
pub fn finalize_snapshot_header_crc32(
    header: &mut LmnkSnapshotHeader,
    container_bytes: &[u8],
) -> LmnkResult<u32> {
    validate_snapshot_header(header, container_bytes.len())?;
    let hdr_size = size_of::<LmnkSnapshotHeader>();
    let crc = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
    header.checksum_crc32 = crc;
    Ok(crc)
}

/// Validates a `.lmnk` snapshot header and verifies the IEEE 802.3 CRC32 checksum over the payload
/// bytes following the 128-byte header (`container_bytes[128..total_size]`).
///
/// # Errors
/// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if header bounds or CRC32 verification fail.
/// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `header.version != LMNK_ABI_VERSION_HEX`.
#[inline]
pub fn verify_snapshot_payload_crc32(
    header: &LmnkSnapshotHeader,
    container_bytes: &[u8],
) -> LmnkResult<()> {
    validate_snapshot_header(header, container_bytes.len())?;
    if header.checksum_crc32 != 0 {
        let hdr_size = size_of::<LmnkSnapshotHeader>();
        let computed = compute_crc32_ieee(&container_bytes[hdr_size..(header.total_size as usize)]);
        if computed != header.checksum_crc32 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
    }
    Ok(())
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
        let header_size = core::mem::size_of::<LmnkSnapshotHeader>() as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(LMNK_ERR_INVALID_ARGUMENT),
        }
    }

    /// Reads and validates a `.lmnk` container's 128-byte [`LmnkSnapshotHeader`] via [`verify_snapshot_payload_crc32`].
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] if `container_bytes` is truncated, magic/offsets
    /// are invalid, or CRC32 verification fails.
    /// Returns [`LMNK_ERR_VERSION_MISMATCH`] if `version != LMNK_ABI_VERSION_HEX`.
    #[inline]
    fn inspect_and_verify_container(
        &self,
        container_bytes: &[u8],
    ) -> LmnkResult<LmnkSnapshotHeader> {
        let header = LmnkSnapshotHeader::read_unaligned_from(container_bytes, 0)
            .map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
        verify_snapshot_payload_crc32(&header, container_bytes)?;
        Ok(header)
    }

    /// Finalizes the IEEE 802.3 CRC32 payload checksum on `header` and writes the 128-byte header
    /// into the beginning of `container_bytes`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_CONTAINER_CORRUPTED`] or [`LMNK_ERR_VERSION_MISMATCH`]
    /// if `header` or `container_bytes` bounds are invalid.
    #[inline]
    fn stamp_and_write_snapshot_header(
        &self,
        header: &mut LmnkSnapshotHeader,
        container_bytes: &mut [u8],
    ) -> LmnkResult<u32> {
        let crc = finalize_snapshot_header_crc32(header, container_bytes)?;
        header
            .write_unaligned_to(container_bytes, 0)
            .map_err(|_| LMNK_ERR_CONTAINER_CORRUPTED)?;
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
        if container_bytes.len() < end {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        Ok(&container_bytes[start..end])
    }

    /// Extracts a hardware-aligned zero-copy [`AlignedSlice`] (`AlignedSlice<'a, T, ALIGN>`) of `elem_count` elements
    /// for `track_id` (`0..=3`) directly from a mapped-storage memory-mapped `.lmnk` container buffer.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `track_id > 3` or `T` is zero-sized.
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `container_bytes` is too small.
    /// Returns [`LMNK_ERR_ALIGNMENT`] if the track start address violates `ALIGN`.
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
/// Zero-Copy In-Memory Mapped Storage (mapped-storage `.lmnk` containers), KV-keyframe keyframe/delta video-stream
/// memory & persona reproduction, and 64-byte phase motion-delta packet dispatch.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedStorageEngine {
    /// Embedded 64-byte C-ABI mapped-storage memory-map & track descriptor (`offset 0..64`).
    pub(crate) descriptor: MappedStoragePreamble,
}

// SAFETY: `MappedStorageEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `MappedStoragePreamble`.
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
            descriptor: MappedStoragePreamble {
                mapped_address: 0,
                file_size_bytes,
                dma_buffer_handle: 0,
                flags,
                fd_handle: -1,
                reserved: [0; 32],
            },
        }
    }

    /// Validates the embedded `MappedStoragePreamble` fields.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `flags == 0` when `file_size_bytes == 0`.
    #[inline(always)]
    pub const fn validate(&self) -> LmnkResult<()> {
        if self.descriptor.file_size_bytes == 0 && self.descriptor.flags == 0 {
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
        let hdr_len = size_of::<LmnkSnapshotHeader>();
        *required_size = hdr_len;
        if buffer.len() < hdr_len {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        let mut hdr = init_snapshot_header(hdr_len as u64, 0);
        let _ = self.stamp_and_write_snapshot_header(&mut hdr, &mut buffer[..hdr_len])?;
        Ok(())
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        _config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()> {
        let header = LmnkSnapshotHeader::read_unaligned_from(buffer, 0)?;
        verify_snapshot_payload_crc32(&header, buffer)?;
        self.descriptor.file_size_bytes = header.total_size;
        Ok(())
    }
}

// ===================================================================
// 8.5. 4-Section `.lmnk` Zero-Copy Container View (`LmnkFourSectionView`)
// ===================================================================

/// Zero-copy view over a 128B-aligned 4-section `.lmnk` memory-mapped container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LmnkFourSectionView<'a> {
    /// Section 0a: Validated 128-byte `.lmnk` snapshot header (`offset 0..128`).
    pub header: &'a LmnkSnapshotHeader,
    /// Section 0b: Crate-private 64-byte section preamble (`offset 128..192`).
    pub(crate) mapped_storage_descriptor: &'a MappedStoragePreamble,
    /// Section 1: Zero-copy slice of crystallized macro attractor centroids (`128B` each).
    pub crystals: &'a [PhaseAnchorRecord],
    /// Section 2: Zero-copy slice of stigmergic pointer trails (`128B` each).
    pub trails: &'a [PointerTrail],
    /// Section 3a: Optional zero-copy reference to the 64-byte Contrastive Phase Bloom Filter.
    pub bloom: Option<&'a ContrastivePhaseBloomFilter>,
    /// Section 3b: Optional zero-copy reference to the 64-byte FastPathLedger telemetry & shortcut slot.
    pub ledger: Option<&'a LedgerSlot>,
    /// Section 3c: Zero-copy slice of 16-byte [`CausalProvenanceReport`] diagnostic history entries.
    pub provenance_reports: &'a [CausalProvenanceReport],
}

impl<'a> LmnkFourSectionView<'a> {
    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections (`num_reports = 0`).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len(
        num_crystals: usize,
        num_trails: usize,
        has_bloom: bool,
        has_ledger: bool,
    ) -> LmnkResult<usize> {
        Self::required_byte_len_with_reports(num_crystals, num_trails, has_bloom, has_ledger, 0)
    }

    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections including `num_reports`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len_with_reports(
        num_crystals: usize,
        num_trails: usize,
        has_bloom: bool,
        has_ledger: bool,
        num_reports: usize,
    ) -> LmnkResult<usize> {
        let off_crystals = LMNK_FOUR_SECTION_HEADER_BYTES;
        let Some(crystals_bytes) = num_crystals.checked_mul(size_of::<PhaseAnchorRecord>()) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let Some(after_crystals) = off_crystals.checked_add(crystals_bytes) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let off_trails = (after_crystals + 127) & !127usize;
        let Some(trails_bytes) = num_trails.checked_mul(size_of::<PointerTrail>()) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let Some(off_bloom) = off_trails.checked_add(trails_bytes) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let bloom_bytes = if has_bloom {
            size_of::<ContrastivePhaseBloomFilter>()
        } else {
            0
        };
        let Some(off_ledger) = off_bloom.checked_add(bloom_bytes) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let ledger_bytes = if has_ledger {
            size_of::<LedgerSlot>()
        } else {
            0
        };
        let Some(off_reports) = off_ledger.checked_add(ledger_bytes) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let Some(reports_bytes) = num_reports.checked_mul(size_of::<CausalProvenanceReport>())
        else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let Some(raw_end) = off_reports.checked_add(reports_bytes) else {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        };
        let min_end = if raw_end < LMNK_FOUR_SECTION_HEADER_BYTES {
            LMNK_FOUR_SECTION_HEADER_BYTES
        } else {
            raw_end
        };
        Ok((min_end + 127) & !127usize)
    }

    /// Serializes `crystals`, `trails`, `bloom_opt`, `ledger_opt`, and `provenance_reports` into `dst_bytes`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len_with_reports(...)`.
    #[inline]
    pub fn write_four_section_with_reports_to_bytes(
        crystals: &[PhaseAnchorRecord],
        trails: &[PointerTrail],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        ledger_opt: Option<&LedgerSlot>,
        provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> LmnkResult<usize> {
        let total_len = Self::required_byte_len_with_reports(
            crystals.len(),
            trails.len(),
            bloom_opt.is_some(),
            ledger_opt.is_some(),
            provenance_reports.len(),
        )?;
        if dst_bytes.len() < total_len {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }

        let buf = &mut dst_bytes[..total_len];
        let off_crystals = LMNK_FOUR_SECTION_HEADER_BYTES;
        let crystals_bytes = PhaseAnchorRecord::write_records_to_container_bytes(
            crystals,
            &mut buf[off_crystals..],
        )?;
        let after_crystals = off_crystals + crystals_bytes;
        let off_trails = (after_crystals + 127) & !127usize;
        let trails_bytes = core::mem::size_of_val(trails);
        let off_bloom = off_trails + trails_bytes;
        let off_ledger = off_bloom
            + if bloom_opt.is_some() {
                size_of::<ContrastivePhaseBloomFilter>()
            } else {
                0
            };
        let off_reports = off_ledger
            + if ledger_opt.is_some() {
                size_of::<LedgerSlot>()
            } else {
                0
            };
        let reports_bytes = core::mem::size_of_val(provenance_reports);
        let end_reports = off_reports + reports_bytes;

        if off_trails > after_crystals {
            buf[after_crystals..off_trails].fill(0);
        }
        PointerTrail::write_unaligned_slice_to(trails, buf, off_trails)?;
        if let Some(bloom) = bloom_opt {
            bloom.write_unaligned_to(buf, off_bloom)?;
        }
        if let Some(ledger) = ledger_opt {
            ledger.write_unaligned_to(buf, off_ledger)?;
        }
        CausalProvenanceReport::write_unaligned_slice_to(provenance_reports, buf, off_reports)?;
        if total_len > end_reports {
            buf[end_reports..total_len].fill(0);
        }
        let storage_desc = MappedStoragePreamble {
            mapped_address: 0,
            file_size_bytes: total_len as u64,
            dma_buffer_handle: 0,
            flags: 4,
            fd_handle: -1,
            reserved: [0; 32],
        };
        storage_desc.write_unaligned_to(buf, LMNK_STORAGE_PREAMBLE_OFFSET)?;

        let checksum_crc32 = Self::compute_container_crc32_with_stamped_crystals(
            buf,
            size_of::<LmnkSnapshotHeader>(),
            off_crystals,
            crystals.len(),
            after_crystals,
            total_len,
        );
        let mut header = init_snapshot_header(total_len as u64, timestamp_epoch_sec);
        header.anchor_offset = off_crystals as u64;
        header.phase_steering_offset = after_crystals as u64;
        header.entropy_offset = off_bloom as u64;
        header.substrate_offset = off_trails as u64;
        header.ledger_slot_offset = off_ledger as u32;
        header.checksum_crc32 = checksum_crc32;
        header.reserved_padding[0] = u8::from(bloom_opt.is_some());
        header.reserved_padding[1] = u8::from(ledger_opt.is_some());
        let rep_cnt = (provenance_reports.len().min(u32::MAX as usize) as u32).to_le_bytes();
        header.reserved_padding[4] = rep_cnt[0];
        header.reserved_padding[5] = rep_cnt[1];
        header.reserved_padding[6] = rep_cnt[2];
        header.reserved_padding[7] = rep_cnt[3];
        header.write_unaligned_to(buf, 0)?;
        Ok(total_len)
    }

    /// Validates and maps a 128-byte aligned 4-section `.lmnk` (mapped-storage) container byte slice
    /// into a zero-copy [`LmnkFourSectionView`] ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_BUFFER_TOO_SMALL`], [`LMNK_ERR_ALIGNMENT`],
    /// [`LMNK_ERR_CONTAINER_CORRUPTED`], or [`LMNK_ERR_VERSION_MISMATCH`]
    /// if the container header, alignment, section offsets, or IEEE 802.3 CRC32 checksums are invalid.
    #[inline]
    pub fn from_container_bytes_zero_copy(bytes: &'a [u8]) -> LmnkResult<Self> {
        if bytes.len() < LMNK_FOUR_SECTION_HEADER_BYTES {
            return Err(LMNK_ERR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(bytes, 128) {
            return Err(LMNK_ERR_ALIGNMENT);
        }

        let header = LmnkSnapshotHeader::from_bytes_at(bytes, 0)?;
        validate_snapshot_header(header, bytes.len())?;
        let mapped_storage_descriptor =
            MappedStoragePreamble::from_bytes_at(bytes, LMNK_STORAGE_PREAMBLE_OFFSET)?;
        if mapped_storage_descriptor.file_size_bytes != 0
            && mapped_storage_descriptor.file_size_bytes != header.total_size
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let total_len = header.total_size as usize;
        let hdr_size = size_of::<LmnkSnapshotHeader>();
        let off_crystals = header.anchor_offset as usize;
        let end_crystals = header.phase_steering_offset as usize;
        let off_trails = header.substrate_offset as usize;
        let off_bloom = header.entropy_offset as usize;
        let off_ledger = header.ledger_slot_offset as usize;

        if off_crystals < LMNK_FOUR_SECTION_HEADER_BYTES
            || off_crystals > end_crystals
            || end_crystals > off_trails
            || off_trails > off_bloom
            || off_bloom > off_ledger
            || off_ledger > total_len
        {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }

        let crystals_byte_len = end_crystals - off_crystals;
        if (crystals_byte_len % size_of::<PhaseAnchorRecord>()) != 0 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let num_crystals = crystals_byte_len / size_of::<PhaseAnchorRecord>();
        let crystals = PhaseAnchorRecord::slice_from_container_bytes_zero_copy(
            &bytes[off_crystals..end_crystals],
            num_crystals,
        )?;
        if header.checksum_crc32 != 0 {
            let computed = Self::compute_container_crc32_with_stamped_crystals(
                bytes,
                hdr_size,
                off_crystals,
                num_crystals,
                end_crystals,
                total_len,
            );
            if computed != header.checksum_crc32 {
                return Err(LMNK_ERR_CONTAINER_CORRUPTED);
            }
        }

        let trails_byte_len = off_bloom - off_trails;
        if (trails_byte_len % size_of::<PointerTrail>()) != 0 {
            return Err(LMNK_ERR_CONTAINER_CORRUPTED);
        }
        let num_trails = trails_byte_len / size_of::<PointerTrail>();
        let trails = PointerTrail::slice_from_bytes_at(bytes, off_trails, num_trails)?;

        // SAFETY: `LmnkSnapshotHeader` is 128-byte aligned (`>= 8`) `#[repr(C)]` POD of 16 `u64` words; offset 9*8 = 72 is in-bounds.
        #[cfg(target_endian = "little")]
        let pad_w0 = unsafe { *core::ptr::from_ref(header).cast::<u64>().add(9) };
        #[cfg(target_endian = "little")]
        let (has_bloom_flag, has_ledger_flag, num_reports) = (
            (pad_w0 & 0xFF) != 0,
            ((pad_w0 >> 8) & 0xFF) != 0,
            (pad_w0 >> 32) as usize,
        );
        #[cfg(not(target_endian = "little"))]
        let (has_bloom_flag, has_ledger_flag, num_reports) = (
            header.reserved_padding[0] != 0,
            header.reserved_padding[1] != 0,
            u32::from_le_bytes([
                header.reserved_padding[4],
                header.reserved_padding[5],
                header.reserved_padding[6],
                header.reserved_padding[7],
            ]) as usize,
        );

        let has_bloom =
            has_bloom_flag & ((off_ledger - off_bloom) >= size_of::<ContrastivePhaseBloomFilter>());
        let bloom = if has_bloom {
            Some(ContrastivePhaseBloomFilter::from_bytes_at(
                bytes, off_bloom,
            )?)
        } else {
            None
        };

        let has_ledger = has_ledger_flag & ((total_len - off_ledger) >= size_of::<LedgerSlot>());
        let ledger = if has_ledger {
            Some(LedgerSlot::from_bytes_at(bytes, off_ledger)?)
        } else {
            None
        };
        let off_reports = off_ledger
            + if has_ledger {
                size_of::<LedgerSlot>()
            } else {
                0
            };
        let provenance_reports = if num_reports > 0 {
            CausalProvenanceReport::slice_from_bytes_at(bytes, off_reports, num_reports)?
        } else {
            &[]
        };

        Ok(Self {
            header,
            mapped_storage_descriptor,
            crystals,
            trails,
            bloom,
            ledger,
            provenance_reports,
        })
    }

    /// Zero-copy extraction of the persisted [`CognitiveTrajectoryLedger`] from `self.ledger`,
    /// returning `Some` when [`LEDGER_FLAG_COGNITIVE_TRAJECTORY`] is active.
    #[inline(always)]
    #[must_use]
    pub const fn cognitive_trajectory_ledger(&self) -> Option<CognitiveTrajectoryLedger> {
        match self.ledger {
            Some(slot) if (slot.auto_tune_flags & LEDGER_FLAG_COGNITIVE_TRAJECTORY) != 0 => {
                Some(CognitiveTrajectoryLedger::from_ledger_slot(slot))
            }
            _ => None,
        }
    }

    #[inline]
    #[must_use]
    fn compute_container_crc32_with_stamped_crystals(
        buf: &[u8],
        hdr_size: usize,
        off_crystals: usize,
        num_crystals: usize,
        after_crystals: usize,
        total_len: usize,
    ) -> u32 {
        #[cfg(target_endian = "little")]
        {
            if is_aligned_to(buf, align_of::<u64>())
                && ((off_crystals - hdr_size) & 63) == 0
                && ((total_len - after_crystals) & 63) == 0
            {
                let mut crc = if (off_crystals - hdr_size) == 64 {
                    // SAFETY: `buf` is 8-byte aligned and `hdr_size..off_crystals` is 64 bytes.
                    let w = unsafe { &*buf.as_ptr().add(hdr_size).cast::<[u64; 8]>() };
                    if ((w[0] | w[2]) == 0)
                        & (w[3] == 0xFFFF_FFFF_0000_0004u64)
                        & (((w[4] | w[5]) | (w[6] | w[7])) == 0)
                    {
                        let fs_lo = w[1] as u32;
                        let fs_hi = (w[1] >> 32) as u32;
                        let fs_c = (if fs_lo == 0 {
                            0
                        } else {
                            crc32_slice16_indep_lo1_u32(fs_lo)
                        }) ^ (if fs_hi == 0 {
                            0
                        } else {
                            crc32_slice16_indep_hi1_u32(fs_hi)
                        });
                        CRC32_MAPPED_STORAGE_DESC_BASE_SEED
                            ^ (if fs_c == 0 { 0 } else { crc32_step_l3(fs_c) })
                    } else {
                        crc32_step_block64_words(0xFFFF_FFFFu32, w)
                    }
                } else {
                    update_aligned64_blocks_crc32_raw(0xFFFF_FFFFu32, &buf[hdr_size..off_crystals])
                };
                let mut idx = 0usize;
                while idx + 4 <= num_crystals {
                    let w6_off0 = off_crystals + idx * size_of::<PhaseAnchorRecord>() + 48;
                    let w6_off1 = w6_off0 + size_of::<PhaseAnchorRecord>();
                    let w6_off2 = w6_off1 + size_of::<PhaseAnchorRecord>();
                    let w6_off3 = w6_off2 + size_of::<PhaseAnchorRecord>();
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off0 + 8 <= total_len`.
                    let rec_crc0 =
                        (unsafe { *buf.as_ptr().add(w6_off0).cast::<u64>() } >> 32) as u32;
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off1 + 8 <= total_len`.
                    let rec_crc1 =
                        (unsafe { *buf.as_ptr().add(w6_off1).cast::<u64>() } >> 32) as u32;
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off2 + 8 <= total_len`.
                    let rec_crc2 =
                        (unsafe { *buf.as_ptr().add(w6_off2).cast::<u64>() } >> 32) as u32;
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off3 + 8 <= total_len`.
                    let rec_crc3 =
                        (unsafe { *buf.as_ptr().add(w6_off3).cast::<u64>() } >> 32) as u32;
                    let s0 = crc32_anchor_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_anchor_stamp_contrib_128b(rec_crc1);
                    let s2 = crc32_anchor_stamp_contrib_128b(rec_crc2);
                    let s3 = crc32_anchor_stamp_contrib_128b(rec_crc3);
                    let p01 = crc32_step_l8(s0) ^ s1;
                    let p23 = crc32_step_l8(s2) ^ s3;
                    crc = crc32_step_l32(crc) ^ crc32_step_l16(p01) ^ p23;
                    idx += 4;
                }
                while idx + 2 <= num_crystals {
                    let w6_off0 = off_crystals + idx * size_of::<PhaseAnchorRecord>() + 48;
                    let w6_off1 = w6_off0 + size_of::<PhaseAnchorRecord>();
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off0 + 8 <= total_len`.
                    let rec_crc0 =
                        (unsafe { *buf.as_ptr().add(w6_off0).cast::<u64>() } >> 32) as u32;
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off1 + 8 <= total_len`.
                    let rec_crc1 =
                        (unsafe { *buf.as_ptr().add(w6_off1).cast::<u64>() } >> 32) as u32;
                    let s0 = crc32_anchor_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_anchor_stamp_contrib_128b(rec_crc1);
                    crc = crc32_step_l16(crc) ^ crc32_step_l8(s0) ^ s1;
                    idx += 2;
                }
                if idx < num_crystals {
                    let w6_off = off_crystals + idx * size_of::<PhaseAnchorRecord>() + 48;
                    // SAFETY: `buf` is >= 8-byte aligned and `w6_off + 8 <= total_len`.
                    let rec_crc = (unsafe { *buf.as_ptr().add(w6_off).cast::<u64>() } >> 32) as u32;
                    crc = crc32_fold_stamped_anchor_128b(crc, rec_crc);
                }
                let mut rem_off = after_crystals;
                if rem_off + 64 <= total_len {
                    // SAFETY: `buf` is 8-byte aligned and `rem_off + 64 <= total_len`.
                    let w_pad = unsafe { &*buf.as_ptr().add(rem_off).cast::<[u64; 8]>() };
                    if (((w_pad[0] | w_pad[1]) | (w_pad[2] | w_pad[3]))
                        | ((w_pad[4] | w_pad[5]) | (w_pad[6] | w_pad[7])))
                        == 0
                    {
                        crc = crc32_step_l4(crc);
                        rem_off += 64;
                    }
                }
                crc = update_aligned64_blocks_crc32_raw(crc, &buf[rem_off..total_len]);
                return !crc;
            }
        }
        compute_aligned64_payload_crc32_ieee(&buf[hdr_size..total_len])
    }
}

const _: () = {
    assert!(size_of::<MappedStorageEngine>() == 64);
    assert!(align_of::<MappedStorageEngine>() == 64);
    assert!(offset_of!(MappedStorageEngine, descriptor) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::AlignedBuffer64;
    use crate::liminal::MAX_ATTENTION_HEADS;
    use crate::runtime::gateway::CAUSAL_AXIS_SUBJECTIVE_PERSONA;
    use crate::stigmergy::ATTRACTOR_SLOT_PERSONA;

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
