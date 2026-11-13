// name: src/core/src/storage.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-Copy In-Memory Mapped Storage (`MappedStorageEngine` 64B `Pod` state descriptor,
//! `GcsoFourSectionView` 4-section `.gcso` container zero-copy persistence & in-place evolution,
//! and PPRC GOP keyframe/delta packets) applying the [`PprcCache`], [`DnpPacketDispatcher`],
//! and [`ZimmsStorage`] traits from [`crate::traits`].

#![deny(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::field_reassign_with_default)]

use crate::abi::{
    gcso_config_t, gcso_daes_slot_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_snapshot_header_t, gcso_zimms_descriptor_t, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
};
use crate::crystal::CrystallizedAttractorRecord;
use crate::traits::aligned::is_aligned_to;
use crate::traits::gateway::{
    compute_aligned64_payload_crc32_ieee, update_aligned64_blocks_crc32_raw,
    CausalProvenanceReport, GatewayFrame, ManifoldGatewaySignal,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, GCSO_FOUR_SECTION_HEADER_BYTES,
    GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET,
};
use crate::traits::primitives::{
    branchless_select_u64, branchless_select_u8, crc32_crystal_stamp_contrib_128b,
    crc32_fold_stamped_crystal_128b, crc32_slice16_indep_hi1_u32, crc32_slice16_indep_lo1_u32,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l3, crc32_step_l32, crc32_step_l4,
    crc32_step_l8, init_snapshot_header, validate_snapshot_header, verify_snapshot_payload_crc32,
    ACTION_MODE_PSPM_ROUTE, CRC32_ZIMMS_DESC_BASE_SEED, DAES_MINI_LEDGER_LEN, DAES_SHORTCUT_SLOTS,
};
pub use crate::traits::primitives::{PprcKeyframePacket, PprcMotionDeltaPacket};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, DAES_METRIC_PHASE_STEERED, DAES_METRIC_PSPM_ROUTED,
    DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE, NUM_SWARM_MICRO_CELLS,
};
use crate::traits::{DnpPacketDispatcher, GcsoResult, Pod, PprcCache, ZimmsStorage};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Zero-Copy In-Memory Mapped Storage (ZIMMS `.gcso` containers), PPRC keyframe/delta video-stream
/// memory & persona reproduction, and DNP 64-byte phase packet dispatch.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedStorageEngine {
    /// Embedded 64-byte C-ABI v0.1.1 ZIMMS memory-map & track descriptor (`offset 0..64`).
    pub descriptor: gcso_zimms_descriptor_t,
}

// SAFETY: `MappedStorageEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `gcso_zimms_descriptor_t`.
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
            descriptor: gcso_zimms_descriptor_t {
                mapped_address: 0,
                file_size_bytes,
                dma_buffer_handle: 0,
                flags,
                fd_handle: -1,
                reserved: [0; 32],
            },
        }
    }

    /// Validates the embedded `gcso_zimms_descriptor_t` fields.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `flags == 0` when `file_size_bytes == 0`.
    #[inline(always)]
    pub const fn validate(&self) -> GcsoResult<()> {
        if self.descriptor.file_size_bytes == 0 && self.descriptor.flags == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }

    /// Verifies and binds an external [`gcso_zimms_descriptor_t`] into this engine.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `desc.file_size_bytes == 0 && desc.flags == 0`.
    #[inline]
    pub fn verify_and_map_container(&mut self, desc: &gcso_zimms_descriptor_t) -> GcsoResult<()> {
        if desc.file_size_bytes == 0 && desc.flags == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.descriptor = *desc;
        Ok(())
    }
}

impl PprcCache for MappedStorageEngine {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> GcsoResult<()> {
        header_out.token_index = token_index;
        header_out.gop_length = 1;
        header_out.composite_vector_length = 1.0;
        Ok(())
    }
}

impl DnpPacketDispatcher for MappedStorageEngine {}

impl ZimmsStorage for MappedStorageEngine {
    #[inline]
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()> {
        let hdr_len = size_of::<gcso_snapshot_header_t>();
        *required_size = hdr_len;
        if buffer.len() < hdr_len {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        let mut hdr = init_snapshot_header(hdr_len as u64, 0);
        let _ = self.stamp_and_write_snapshot_header(&mut hdr, &mut buffer[..hdr_len])?;
        Ok(())
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        _config_out: &mut gcso_config_t,
    ) -> GcsoResult<()> {
        let header = gcso_snapshot_header_t::read_unaligned_from(buffer, 0)?;
        verify_snapshot_payload_crc32(&header, buffer)?;
        self.descriptor.file_size_bytes = header.total_size;
        Ok(())
    }

    #[inline]
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_zimms_descriptor_t,
    ) -> GcsoResult<()> {
        if file_path.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *descriptor_out = self.descriptor;
        Ok(())
    }

    #[inline]
    fn close_mmap(&mut self, descriptor: &mut gcso_zimms_descriptor_t) -> GcsoResult<()> {
        descriptor.mapped_address = 0;
        descriptor.fd_handle = -1;
        self.descriptor = *descriptor;
        Ok(())
    }
}

// ===================================================================
// 8.5. 4-Section `.gcso` (ZIMMS) Zero-Copy Container View (`GcsoFourSectionView`)
// ===================================================================

/// Zero-copy view over a 128B-aligned 4-section `.gcso` (ZIMMS) memory-mapped container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcsoFourSectionView<'a> {
    /// Section 0a: Validated 128-byte `.gcso` snapshot header (`offset 0..128`).
    pub header: &'a gcso_snapshot_header_t,
    /// Section 0b: Validated 64-byte ZIMMS memory-map & track descriptor (`offset 128..192`).
    pub zimms_descriptor: &'a gcso_zimms_descriptor_t,
    /// Section 1: Zero-copy slice of crystallized macro attractor centroids (`128B` each).
    pub crystals: &'a [CrystallizedAttractorRecord],
    /// Section 2: Zero-copy slice of stigmergic pointer trails (`128B` each).
    pub trails: &'a [gcso_pointer_trail_t],
    /// Section 3a: Optional zero-copy reference to the 64-byte Contrastive Phase Bloom Filter.
    pub bloom: Option<&'a ContrastivePhaseBloomFilter>,
    /// Section 3b: Optional zero-copy reference to the 64-byte DAES telemetry & shortcut slot.
    pub daes: Option<&'a gcso_daes_slot_t>,
    /// Section 3c: Zero-copy slice of 16-byte [`CausalProvenanceReport`] diagnostic history entries.
    pub provenance_reports: &'a [CausalProvenanceReport],
}

impl<'a> GcsoFourSectionView<'a> {
    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections (`num_reports = 0`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len(
        num_crystals: usize,
        num_trails: usize,
        has_bloom: bool,
        has_daes: bool,
    ) -> GcsoResult<usize> {
        Self::required_byte_len_with_reports(num_crystals, num_trails, has_bloom, has_daes, 0)
    }

    /// Computes the exact 128-byte-aligned container byte length required for the 4 sections including `num_reports`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] on arithmetic overflow.
    #[inline]
    pub const fn required_byte_len_with_reports(
        num_crystals: usize,
        num_trails: usize,
        has_bloom: bool,
        has_daes: bool,
        num_reports: usize,
    ) -> GcsoResult<usize> {
        let off_crystals = GCSO_FOUR_SECTION_HEADER_BYTES;
        let Some(crystals_bytes) =
            num_crystals.checked_mul(size_of::<CrystallizedAttractorRecord>())
        else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(after_crystals) = off_crystals.checked_add(crystals_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let off_trails = (after_crystals + 127) & !127usize;
        let Some(trails_bytes) = num_trails.checked_mul(size_of::<gcso_pointer_trail_t>()) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(off_bloom) = off_trails.checked_add(trails_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let bloom_bytes = if has_bloom {
            size_of::<ContrastivePhaseBloomFilter>()
        } else {
            0
        };
        let Some(off_daes) = off_bloom.checked_add(bloom_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let daes_bytes = if has_daes {
            size_of::<gcso_daes_slot_t>()
        } else {
            0
        };
        let Some(off_reports) = off_daes.checked_add(daes_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(reports_bytes) = num_reports.checked_mul(size_of::<CausalProvenanceReport>())
        else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let Some(raw_end) = off_reports.checked_add(reports_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let min_end = if raw_end < GCSO_FOUR_SECTION_HEADER_BYTES {
            GCSO_FOUR_SECTION_HEADER_BYTES
        } else {
            raw_end
        };
        Ok((min_end + 127) & !127usize)
    }

    /// Serializes `crystals`, `trails`, `bloom_opt`, and `daes_opt` into `dst_bytes` and stamps IEEE 802.3 CRC32s.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len(...)`.
    #[inline]
    pub fn write_four_section_container_to_bytes(
        crystals: &[CrystallizedAttractorRecord],
        trails: &[gcso_pointer_trail_t],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        daes_opt: Option<&gcso_daes_slot_t>,
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> GcsoResult<usize> {
        Self::write_four_section_with_reports_to_bytes(
            crystals,
            trails,
            bloom_opt,
            daes_opt,
            &[],
            timestamp_epoch_sec,
            dst_bytes,
        )
    }

    /// Serializes `crystals`, `trails`, `bloom_opt`, `daes_opt`, and `provenance_reports` into `dst_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len_with_reports(...)`.
    #[inline]
    pub fn write_four_section_with_reports_to_bytes(
        crystals: &[CrystallizedAttractorRecord],
        trails: &[gcso_pointer_trail_t],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        daes_opt: Option<&gcso_daes_slot_t>,
        provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> GcsoResult<usize> {
        let total_len = Self::required_byte_len_with_reports(
            crystals.len(),
            trails.len(),
            bloom_opt.is_some(),
            daes_opt.is_some(),
            provenance_reports.len(),
        )?;
        if dst_bytes.len() < total_len {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        let buf = &mut dst_bytes[..total_len];
        let off_crystals = GCSO_FOUR_SECTION_HEADER_BYTES;
        let crystals_bytes = CrystallizedAttractorRecord::write_records_to_zimms_bytes(
            crystals,
            &mut buf[off_crystals..],
        )?;
        let after_crystals = off_crystals + crystals_bytes;
        let off_trails = (after_crystals + 127) & !127usize;
        let trails_bytes = core::mem::size_of_val(trails);
        let off_bloom = off_trails + trails_bytes;
        let off_daes = off_bloom
            + if bloom_opt.is_some() {
                size_of::<ContrastivePhaseBloomFilter>()
            } else {
                0
            };
        let off_reports = off_daes
            + if daes_opt.is_some() {
                size_of::<gcso_daes_slot_t>()
            } else {
                0
            };
        let reports_bytes = core::mem::size_of_val(provenance_reports);
        let end_reports = off_reports + reports_bytes;

        if off_trails > after_crystals {
            buf[after_crystals..off_trails].fill(0);
        }
        gcso_pointer_trail_t::write_unaligned_slice_to(trails, buf, off_trails)?;
        if let Some(bloom) = bloom_opt {
            bloom.write_unaligned_to(buf, off_bloom)?;
        }
        if let Some(daes) = daes_opt {
            daes.write_unaligned_to(buf, off_daes)?;
        }
        CausalProvenanceReport::write_unaligned_slice_to(provenance_reports, buf, off_reports)?;
        if total_len > end_reports {
            buf[end_reports..total_len].fill(0);
        }
        let zimms_desc = gcso_zimms_descriptor_t {
            mapped_address: 0,
            file_size_bytes: total_len as u64,
            dma_buffer_handle: 0,
            flags: 4,
            fd_handle: -1,
            reserved: [0; 32],
        };
        zimms_desc.write_unaligned_to(buf, GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET)?;

        let checksum_crc32 = Self::compute_container_crc32_with_stamped_crystals(
            buf,
            size_of::<gcso_snapshot_header_t>(),
            off_crystals,
            crystals.len(),
            after_crystals,
            total_len,
        );
        let mut header = init_snapshot_header(total_len as u64, timestamp_epoch_sec);
        header.attractor_field_offset = off_crystals as u64;
        header.dpsr_state_offset = after_crystals as u64;
        header.edbc_state_offset = off_bloom as u64;
        header.action_hub_offset = off_trails as u64;
        header.daes_slot_offset = off_daes as u32;
        header.checksum_crc32 = checksum_crc32;
        header.reserved_padding[0] = u8::from(bloom_opt.is_some());
        header.reserved_padding[1] = u8::from(daes_opt.is_some());
        let rep_cnt = (provenance_reports.len().min(u32::MAX as usize) as u32).to_le_bytes();
        header.reserved_padding[4] = rep_cnt[0];
        header.reserved_padding[5] = rep_cnt[1];
        header.reserved_padding[6] = rep_cnt[2];
        header.reserved_padding[7] = rep_cnt[3];
        header.write_unaligned_to(buf, 0)?;
        Ok(total_len)
    }

    /// Validates and maps a 128-byte aligned 4-section `.gcso` (ZIMMS) container byte slice
    /// into a zero-copy [`GcsoFourSectionView`] ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`], or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the container header, alignment, section offsets, or IEEE 802.3 CRC32 checksums are invalid.
    #[inline]
    pub fn from_zimms_bytes_zero_copy(bytes: &'a [u8]) -> GcsoResult<Self> {
        if bytes.len() < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(bytes, 128) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }

        let header = gcso_snapshot_header_t::from_bytes_at(bytes, 0)?;
        validate_snapshot_header(header, bytes.len())?;
        let zimms_descriptor =
            gcso_zimms_descriptor_t::from_bytes_at(bytes, GCSO_ZIMMS_DESCRIPTOR_PREAMBLE_OFFSET)?;
        if zimms_descriptor.file_size_bytes != 0
            && zimms_descriptor.file_size_bytes != header.total_size
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let total_len = header.total_size as usize;
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let off_daes = header.daes_slot_offset as usize;

        if off_crystals < GCSO_FOUR_SECTION_HEADER_BYTES
            || off_crystals > end_crystals
            || end_crystals > off_trails
            || off_trails > off_bloom
            || off_bloom > off_daes
            || off_daes > total_len
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let crystals_byte_len = end_crystals - off_crystals;
        if (crystals_byte_len % size_of::<CrystallizedAttractorRecord>()) != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let num_crystals = crystals_byte_len / size_of::<CrystallizedAttractorRecord>();
        let crystals = CrystallizedAttractorRecord::slice_from_zimms_bytes_zero_copy(
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
                return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
            }
        }

        let trails_byte_len = off_bloom - off_trails;
        if (trails_byte_len % size_of::<gcso_pointer_trail_t>()) != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let num_trails = trails_byte_len / size_of::<gcso_pointer_trail_t>();
        let trails = gcso_pointer_trail_t::slice_from_bytes_at(bytes, off_trails, num_trails)?;

        #[cfg(target_endian = "little")]
        let pad_w0 = unsafe { *core::ptr::from_ref(header).cast::<u64>().add(9) };
        #[cfg(target_endian = "little")]
        let (has_bloom_flag, has_daes_flag, num_reports) = (
            (pad_w0 & 0xFF) != 0,
            ((pad_w0 >> 8) & 0xFF) != 0,
            (pad_w0 >> 32) as usize,
        );
        #[cfg(not(target_endian = "little"))]
        let (has_bloom_flag, has_daes_flag, num_reports) = (
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
            has_bloom_flag & ((off_daes - off_bloom) >= size_of::<ContrastivePhaseBloomFilter>());
        let bloom = if has_bloom {
            Some(ContrastivePhaseBloomFilter::from_bytes_at(
                bytes, off_bloom,
            )?)
        } else {
            None
        };

        let has_daes = has_daes_flag & ((total_len - off_daes) >= size_of::<gcso_daes_slot_t>());
        let daes = if has_daes {
            Some(gcso_daes_slot_t::from_bytes_at(bytes, off_daes)?)
        } else {
            None
        };
        let off_reports = off_daes
            + if has_daes {
                size_of::<gcso_daes_slot_t>()
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
            zimms_descriptor,
            crystals,
            trails,
            bloom,
            daes,
            provenance_reports,
        })
    }

    #[inline(always)]
    pub(crate) fn validate_four_section_mut_offsets(
        zimms_bytes: &[u8],
    ) -> GcsoResult<(
        gcso_snapshot_header_t,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )> {
        if zimms_bytes.len() < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(zimms_bytes, 128) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }

        let header = *gcso_snapshot_header_t::from_bytes_at(zimms_bytes, 0)?;
        validate_snapshot_header(&header, zimms_bytes.len())?;
        let total_len = header.total_size as usize;
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let off_daes = header.daes_slot_offset as usize;

        if off_crystals < GCSO_FOUR_SECTION_HEADER_BYTES
            || off_crystals > end_crystals
            || end_crystals > off_trails
            || off_trails > off_bloom
            || off_bloom > off_daes
            || off_daes > total_len
            || (off_crystals % align_of::<CrystallizedAttractorRecord>()) != 0
            || (off_trails % align_of::<gcso_pointer_trail_t>()) != 0
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let crystals_bytes = end_crystals - off_crystals;
        let trails_bytes = off_bloom - off_trails;
        if (crystals_bytes % size_of::<CrystallizedAttractorRecord>()) != 0
            || (trails_bytes % size_of::<gcso_pointer_trail_t>()) != 0
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let num_crystals = crystals_bytes / size_of::<CrystallizedAttractorRecord>();
        let num_trails = trails_bytes / size_of::<gcso_pointer_trail_t>();
        Ok((
            header,
            total_len,
            off_crystals,
            num_crystals,
            off_trails,
            num_trails,
            off_bloom,
            off_daes,
        ))
    }

    #[inline(always)]
    pub(crate) fn update_four_section_daes_in_place(
        zimms_bytes: &mut [u8],
        header: &gcso_snapshot_header_t,
        total_len: usize,
        off_daes: usize,
        metric_code: u8,
        shortcut_ptr: u64,
    ) -> GcsoResult<bool> {
        let has_daes = (header.reserved_padding[1] != 0)
            & ((total_len - off_daes) >= size_of::<gcso_daes_slot_t>());
        if has_daes {
            let daes_mut = gcso_daes_slot_t::from_bytes_mut_at(zimms_bytes, off_daes)?;
            let ring_head = (daes_mut.telemetry_ring_head as usize) & (DAES_MINI_LEDGER_LEN - 1);
            daes_mut.telemetry_mini_ledger[ring_head] = metric_code;
            daes_mut.telemetry_ring_head = daes_mut.telemetry_ring_head.wrapping_add(1);
            if shortcut_ptr != 0 {
                let sc_idx = ((shortcut_ptr.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62) as usize;
                daes_mut.fast_path_shortcuts[sc_idx & (DAES_SHORTCUT_SLOTS - 1)] = shortcut_ptr;
                daes_mut.fast_path_bypass_mask |= shortcut_ptr;
            }
        }
        Ok(has_daes)
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
                        CRC32_ZIMMS_DESC_BASE_SEED
                            ^ (if fs_c == 0 { 0 } else { crc32_step_l3(fs_c) })
                    } else {
                        crc32_step_block64_words(0xFFFF_FFFFu32, w)
                    }
                } else {
                    update_aligned64_blocks_crc32_raw(0xFFFF_FFFFu32, &buf[hdr_size..off_crystals])
                };
                let mut idx = 0usize;
                while idx + 4 <= num_crystals {
                    let w6_off0 =
                        off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 48;
                    let w6_off1 = w6_off0 + size_of::<CrystallizedAttractorRecord>();
                    let w6_off2 = w6_off1 + size_of::<CrystallizedAttractorRecord>();
                    let w6_off3 = w6_off2 + size_of::<CrystallizedAttractorRecord>();
                    let rec_crc0 =
                        (unsafe { *buf.as_ptr().add(w6_off0).cast::<u64>() } >> 32) as u32;
                    let rec_crc1 =
                        (unsafe { *buf.as_ptr().add(w6_off1).cast::<u64>() } >> 32) as u32;
                    let rec_crc2 =
                        (unsafe { *buf.as_ptr().add(w6_off2).cast::<u64>() } >> 32) as u32;
                    let rec_crc3 =
                        (unsafe { *buf.as_ptr().add(w6_off3).cast::<u64>() } >> 32) as u32;
                    let s0 = crc32_crystal_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_crystal_stamp_contrib_128b(rec_crc1);
                    let s2 = crc32_crystal_stamp_contrib_128b(rec_crc2);
                    let s3 = crc32_crystal_stamp_contrib_128b(rec_crc3);
                    let p01 = crc32_step_l8(s0) ^ s1;
                    let p23 = crc32_step_l8(s2) ^ s3;
                    crc = crc32_step_l32(crc) ^ crc32_step_l16(p01) ^ p23;
                    idx += 4;
                }
                while idx + 2 <= num_crystals {
                    let w6_off0 =
                        off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 48;
                    let w6_off1 = w6_off0 + size_of::<CrystallizedAttractorRecord>();
                    let rec_crc0 =
                        (unsafe { *buf.as_ptr().add(w6_off0).cast::<u64>() } >> 32) as u32;
                    let rec_crc1 =
                        (unsafe { *buf.as_ptr().add(w6_off1).cast::<u64>() } >> 32) as u32;
                    let s0 = crc32_crystal_stamp_contrib_128b(rec_crc0);
                    let s1 = crc32_crystal_stamp_contrib_128b(rec_crc1);
                    crc = crc32_step_l16(crc) ^ crc32_step_l8(s0) ^ s1;
                    idx += 2;
                }
                if idx < num_crystals {
                    let w6_off = off_crystals + idx * size_of::<CrystallizedAttractorRecord>() + 48;
                    let rec_crc = (unsafe { *buf.as_ptr().add(w6_off).cast::<u64>() } >> 32) as u32;
                    crc = crc32_fold_stamped_crystal_128b(crc, rec_crc);
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

    #[inline(always)]
    pub(crate) fn finalize_four_section_header_in_place(
        zimms_bytes: &mut [u8],
        header: &mut gcso_snapshot_header_t,
        total_len: usize,
    ) -> GcsoResult<()> {
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let num_crystals = (end_crystals - off_crystals) / size_of::<CrystallizedAttractorRecord>();
        let crc = Self::compute_container_crc32_with_stamped_crystals(
            zimms_bytes,
            hdr_size,
            off_crystals,
            num_crystals,
            end_crystals,
            total_len,
        );
        header.checksum_crc32 = crc;
        let crc_off = offset_of!(gcso_snapshot_header_t, checksum_crc32);
        zimms_bytes[crc_off..crc_off + 4].copy_from_slice(&crc.to_le_bytes());
        Ok(())
    }

    /// **Zero-Copy In-Place `.gcso` 4-Section Container Evolution**:
    /// Mutates a 128-byte aligned `.gcso` (ZIMMS) container buffer (`zimms_bytes`) directly in-place:
    /// 1. Validates Section 0 (`gcso_snapshot_header_t`) and section offsets.
    /// 2. Synthesizes `signal` via [`ManifoldGatewaySignal::synthesize_gateway_frame_with_words`]
    ///    and executes Online Toroidal $k$-Means / DP-Means (`step_online_toroidal_kmeans_frame_with_words_64`)
    ///    directly on Section 1 (`&mut [CrystallizedAttractorRecord]`) and Section 2 (`trails.get_mut(trail_idx)`).
    /// 3. Optionally consolidates redundant centroids when `merge_coherence_thresh < 1.0` via
    ///    [`CrystallizedAttractorRecord::merge_and_prune_centroids_64`].
    /// 4. If Section 3a (`bloom`) is present and `frame` triggered repulsion, records the anti-phase
    ///    pattern in-place; if Section 3c (`provenance_reports`) has capacity, writes `frame.to_causal_report()`
    ///    into the circular diagnostic history slot in-place.
    /// 5. Recomputes and stamps `header.checksum_crc32` in-place.
    ///
    /// Returns `(GatewayFrame, assigned_cluster_idx, assigned_anchor_id, updated_gravity_mass, spawned_new_cluster, merged_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`], or
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `zimms_bytes` fails layout or alignment checks.
    #[inline]
    pub fn evolve_four_section_kmeans_in_place<S: ManifoldGatewaySignal + ?Sized>(
        zimms_bytes: &mut [u8],
        signal: &S,
        split_coherence_thresh: f32,
        merge_coherence_thresh: f32,
        trail_idx: usize,
    ) -> GcsoResult<(GatewayFrame, usize, u32, f32, bool, u32)> {
        let (
            mut header,
            total_len,
            off_crystals,
            num_crystals,
            off_trails,
            num_trails,
            off_bloom,
            off_daes,
        ) = Self::validate_four_section_mut_offsets(zimms_bytes)?;

        let (frame, sample_w) = signal.synthesize_gateway_frame_with_words();
        let (crystals_mut, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_mut_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (c_idx, c_id, _coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                crystals_mut,
                &frame,
                &sample_w,
                split_coherence_thresh,
                trails_mut.get_mut(trail_idx),
            );
        let merged = if merge_coherence_thresh < 1.0 && crystals_mut.len() >= 2 {
            CrystallizedAttractorRecord::merge_and_prune_centroids_64(
                crystals_mut,
                merge_coherence_thresh,
                0.0,
                1.0,
            )
            .0
        } else {
            0
        };

        let rep = frame.to_causal_report();
        let has_bloom = (header.reserved_padding[0] != 0)
            & ((off_daes - off_bloom) >= size_of::<ContrastivePhaseBloomFilter>());
        if has_bloom && rep.is_cohomology_repulsion() {
            let bloom_mut = ContrastivePhaseBloomFilter::from_bytes_mut_at(zimms_bytes, off_bloom)?;
            bloom_mut.insert_repulsive_pattern_words_weighted(&sample_w, 2);
        }

        let metric_code = branchless_select_u8(
            rep.is_cohomology_repulsion(),
            DAES_METRIC_REPULSION_APPLIED,
            DAES_METRIC_PHASE_STEERED,
        ) | branchless_select_u8(
            (frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0,
            DAES_METRIC_PSPM_ROUTED,
            0,
        ) | branchless_select_u8(
            (frame.provenance_bits & CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED) != 0,
            DAES_METRIC_SLOT_SYNERGY_ACTIVE,
            0,
        );
        let shortcut_ptr =
            branchless_select_u64(rep.is_cohomology_repulsion(), 0, frame.tagged_ptr);
        let has_daes = Self::update_four_section_daes_in_place(
            zimms_bytes,
            &header,
            total_len,
            off_daes,
            metric_code,
            shortcut_ptr,
        )?;

        let num_reports = u32::from_le_bytes([
            header.reserved_padding[4],
            header.reserved_padding[5],
            header.reserved_padding[6],
            header.reserved_padding[7],
        ]) as usize;
        if num_reports > 0 {
            let off_reports = off_daes
                + if has_daes {
                    size_of::<gcso_daes_slot_t>()
                } else {
                    0
                };
            let rep_end = off_reports
                .checked_add(num_reports * size_of::<CausalProvenanceReport>())
                .ok_or(GCSO_ERROR_CONTAINER_CORRUPTED)?;
            if rep_end <= total_len {
                let write_cursor = u32::from_le_bytes([
                    header.reserved_padding[8],
                    header.reserved_padding[9],
                    header.reserved_padding[10],
                    header.reserved_padding[11],
                ]) as usize;
                let slot_idx = write_cursor % num_reports;
                let slot_off = off_reports + slot_idx * size_of::<CausalProvenanceReport>();
                rep.write_unaligned_to(zimms_bytes, slot_off)?;
                let next_cursor = ((write_cursor.wrapping_add(1)) as u32).to_le_bytes();
                header.reserved_padding[8] = next_cursor[0];
                header.reserved_padding[9] = next_cursor[1];
                header.reserved_padding[10] = next_cursor[2];
                header.reserved_padding[11] = next_cursor[3];
                let cursor_off = offset_of!(gcso_snapshot_header_t, reserved_padding) + 8;
                zimms_bytes[cursor_off..cursor_off + 4].copy_from_slice(&next_cursor);
            }
        }

        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((frame, c_idx, c_id, mass, spawned, merged))
    }

    #[inline(always)]
    pub(crate) fn route_four_section_coarse_to_fine_with_words_in_place(
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
    ) -> GcsoResult<(usize, u32, f32, f32, [u64; NUM_SWARM_MICRO_CELLS])> {
        let (
            mut header,
            total_len,
            off_crystals,
            num_crystals,
            off_trails,
            num_trails,
            _off_bloom,
            _off_daes,
        ) = Self::validate_four_section_mut_offsets(zimms_bytes)?;
        if trail_idx >= num_trails {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (crystals_ref, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (best_idx, anchor_id, best_coh, best_grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                crystals_ref,
                &mut trails_mut[trail_idx],
                pull_step_u7,
            );
        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((best_idx, anchor_id, best_coh, best_grav, bridged_w))
    }

    /// **Zero-Copy In-Place Coarse-to-Fine Trail Routing on a 4-Section `.gcso` Container**:
    /// Routes `trails[trail_idx]` inside Section 2 against the crystallized macro centroids in
    /// Section 1 (`crystals`) via [`CrystallizedAttractorRecord::route_coarse_to_fine_trail_64`]
    /// and updates `header.checksum_crc32` in-place.
    ///
    /// Returns `(selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_INVALID_ARGUMENT`], or [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn route_four_section_coarse_to_fine_in_place(
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
    ) -> GcsoResult<(usize, u32, f32, f32)> {
        let (best_idx, anchor_id, best_coh, best_grav, _bridged_w) =
            Self::route_four_section_coarse_to_fine_with_words_in_place(
                zimms_bytes,
                trail_idx,
                pull_step_u7,
            )?;
        Ok((best_idx, anchor_id, best_coh, best_grav))
    }

    /// Routes `trails[trail_idx]` in-place against the top-2 macro centroids in `zimms_bytes`
    /// and returns both the routing metadata and the 3 register-resident 8-word phase arrays
    /// `(raw_w, fact_w, logic_w)` together with `(header, total_len, off_daes, updated_ptr)` so
    /// optional Q/K steering callers can fuse their DAES telemetry bits before finalizing the header.
    #[inline(always)]
    pub(crate) fn route_four_section_top2_liminal_core_in_place(
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> GcsoResult<(
        usize,
        usize,
        u32,
        u32,
        f32,
        f32,
        bool,
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
        [u64; NUM_SWARM_MICRO_CELLS],
        gcso_snapshot_header_t,
        usize,
        usize,
        u64,
    )> {
        let (
            header,
            total_len,
            off_crystals,
            num_crystals,
            off_trails,
            num_trails,
            _off_bloom,
            off_daes,
        ) = Self::validate_four_section_mut_offsets(zimms_bytes)?;
        if trail_idx >= num_trails {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (crystals_ref, trails_mut) = {
            let (head_bytes, tail_bytes) = zimms_bytes.split_at_mut(off_trails);
            let crystals = CrystallizedAttractorRecord::slice_from_bytes_at(
                head_bytes,
                off_crystals,
                num_crystals,
            )?;
            let trails = gcso_pointer_trail_t::slice_from_bytes_mut_at(tail_bytes, 0, num_trails)?;
            (crystals, trails)
        };

        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, raw_w, fact_w, logic_w) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                crystals_ref,
                &mut trails_mut[trail_idx],
                pull_step_u7,
                liminal_gravity_ratio_thresh,
            );
        let updated_ptr = trails_mut[trail_idx].current_ptr;
        Ok((
            idx1,
            idx2,
            aid1,
            aid2,
            eff_coh,
            tot_grav,
            is_liminal,
            raw_w,
            fact_w,
            logic_w,
            header,
            total_len,
            off_daes,
            updated_ptr,
        ))
    }

    /// **Zero-Copy In-Place Top-2 Liminal Bifurcation Routing on a 4-Section `.gcso` Container**:
    /// Routes `trails[trail_idx]` inside Section 2 against the Top-2 nearest macro centroids in
    /// Section 1 (`crystals`) via [`CrystallizedAttractorRecord::route_top2_liminal_centroids_64`],
    /// updates Section 3b (`daes`) telemetry if present, and stamps `header.checksum_crc32` in-place.
    ///
    /// Returns `(primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_INVALID_ARGUMENT`], or [`GCSO_ERROR_CONTAINER_CORRUPTED`] on validation failure.
    #[inline]
    pub fn route_four_section_top2_liminal_in_place(
        zimms_bytes: &mut [u8],
        trail_idx: usize,
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
    ) -> GcsoResult<(usize, usize, u32, u32, f32, f32, bool)> {
        let (
            idx1,
            idx2,
            aid1,
            aid2,
            eff_coh,
            tot_grav,
            is_liminal,
            _raw_w,
            _fact_w,
            _logic_w,
            mut header,
            total_len,
            off_daes,
            updated_ptr,
        ) = Self::route_four_section_top2_liminal_core_in_place(
            zimms_bytes,
            trail_idx,
            pull_step_u7,
            liminal_gravity_ratio_thresh,
        )?;
        let metric_code =
            branchless_select_u8((aid1 != 0) | is_liminal, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        let _ = Self::update_four_section_daes_in_place(
            zimms_bytes,
            &header,
            total_len,
            off_daes,
            metric_code,
            updated_ptr,
        )?;
        Self::finalize_four_section_header_in_place(zimms_bytes, &mut header, total_len)?;
        Ok((idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal))
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
    use crate::abi::{gcso_daes_slot_t, gcso_pointer_trail_t, gcso_tagged_ptr_t};
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::{ATTRACTOR_SLOT_PERSONA, MAX_ATTENTION_HEADS};
    use crate::traits::CAUSAL_AXIS_SUBJECTIVE_PERSONA;

    #[test]
    fn test_mapped_storage_engine_dnp_and_pprc_gop_video_stream_seek() {
        let engine = MappedStorageEngine::new(4096, 1);
        assert_eq!(engine.validate(), Ok(()));

        let daes_slot = gcso_daes_slot_t {
            mode: 1,
            ..Default::default()
        };
        let mut packet_buf = AlignedBuffer64::new([0u8; 64]);
        let phases = [12i8; MAX_ATTENTION_HEADS];
        let crc = engine
            .encode_dnp_phase_packet(
                &daes_slot,
                2,
                0x09,
                &phases,
                &mut packet_buf.as_aligned_slice_mut(),
            )
            .unwrap();
        let mut trail = gcso_pointer_trail_t::default();
        let (slot, mode, is_repulsion) = engine
            .decode_and_apply_dnp_packet_to_trail(
                &daes_slot,
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

        let rep_crc = engine
            .encode_dnp_phase_packet(
                &daes_slot,
                3,
                0x05,
                &phases,
                &mut packet_buf.as_aligned_slice_mut(),
            )
            .unwrap();
        let mut rep_trail = gcso_pointer_trail_t::default();
        let (rep_slot, rep_mode, rep_is_repulsion) = engine
            .decode_and_apply_dnp_packet_to_trail(
                &daes_slot,
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

        // Verify PPRC Video-Codec Style I-Frame / P-Frame GOP Stream & Zero-Forward Latency Seek
        let anchor_t0 = [16i8; MAX_ATTENTION_HEADS];
        let state_t1 = [20i8; MAX_ATTENTION_HEADS];
        let state_t2 = [26i8; MAX_ATTENTION_HEADS];

        let iframe = engine.encode_pprc_iframe_packet(
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
            .encode_pprc_pframe_delta_packet(&anchor_t0, &state_t1, 1)
            .unwrap();
        let pframe2 = engine
            .encode_pprc_pframe_delta_packet(&state_t1, &state_t2, 1)
            .unwrap();
        let gop_pframes = [pframe1, pframe2];

        let mut stream_bytes = [0u8; 256];
        let written = engine
            .pack_pprc_gop_stream_bytes(&iframe, &gop_pframes, &mut stream_bytes)
            .unwrap();
        assert_eq!(written, 256);

        let mut seek_trail = gcso_pointer_trail_t::default();
        let (lp, r_bar, kappa, v_shift, rep_cnt) = engine
            .unpack_and_seek_pprc_gop_stream_bytes(
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
        assert_eq!(gcso_tagged_ptr_t(seek_trail.current_ptr).unpack_addr(), 102);
        assert!(lp > -0.05 && (r_bar - 1.0).abs() < 1.0e-4 && kappa > 10.0 && v_shift > 0.0);
    }
}
