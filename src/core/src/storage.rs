// name: src/core/src/storage.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Zero-Copy In-Memory Mapped Storage (`MappedStorageEngine` 64B `Pod` state descriptor,
//! `GcsoFourSectionView` 4-section `.gcso` container zero-copy persistence & in-place evolution,
//! and KV-keyframe GOP keyframe/delta packets) applying the [`KvKeyframeCache`], [`DnpPacketDispatcher`],
//! and [`MappedStorage`] traits from [`crate::traits`].

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
    gcso_config_t, gcso_kv_keyframe_header_t, gcso_ledger_slot_t, gcso_mapped_storage_descriptor_t,
    gcso_pointer_trail_t, gcso_snapshot_header_t, GCSO_ERROR_BUFFER_TOO_SMALL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
};
use crate::crystal::CrystallizedAttractorRecord;
use crate::traits::aligned::is_aligned_to;
use crate::traits::gateway::CognitiveTrajectoryLedger;
use crate::traits::gateway::{
    compute_aligned64_payload_crc32_ieee, update_aligned64_blocks_crc32_raw,
    CausalProvenanceReport, GCSO_FOUR_SECTION_HEADER_BYTES, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET,
};
use crate::traits::primitives::{
    crc32_crystal_stamp_contrib_128b, crc32_fold_stamped_crystal_128b, crc32_slice16_indep_hi1_u32,
    crc32_slice16_indep_lo1_u32, crc32_step_block64_words, crc32_step_l16, crc32_step_l3,
    crc32_step_l32, crc32_step_l4, crc32_step_l8, init_snapshot_header, validate_snapshot_header,
    verify_snapshot_payload_crc32, CRC32_MAPPED_STORAGE_DESC_BASE_SEED,
};
pub use crate::traits::primitives::{KvKeyframePacket, KvMotionDeltaPacket};
use crate::traits::telemetry::{ContrastivePhaseBloomFilter, LEDGER_FLAG_COGNITIVE_TRAJECTORY};
use crate::traits::{DnpPacketDispatcher, GcsoResult, KvKeyframeCache, MappedStorage, Pod};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Zero-Copy In-Memory Mapped Storage (mapped-storage `.gcso` containers), KV-keyframe keyframe/delta video-stream
/// memory & persona reproduction, and DNP 64-byte phase packet dispatch.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MappedStorageEngine {
    /// Embedded 64-byte C-ABI mapped-storage memory-map & track descriptor (`offset 0..64`).
    pub descriptor: gcso_mapped_storage_descriptor_t,
}

// SAFETY: `MappedStorageEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `gcso_mapped_storage_descriptor_t`.
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
            descriptor: gcso_mapped_storage_descriptor_t {
                mapped_address: 0,
                file_size_bytes,
                dma_buffer_handle: 0,
                flags,
                fd_handle: -1,
                reserved: [0; 32],
            },
        }
    }

    /// Validates the embedded `gcso_mapped_storage_descriptor_t` fields.
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
}

impl KvKeyframeCache for MappedStorageEngine {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> GcsoResult<()> {
        header_out.token_index = token_index;
        header_out.gop_length = 1;
        header_out.composite_vector_length = 1.0;
        Ok(())
    }
}

impl DnpPacketDispatcher for MappedStorageEngine {}

impl MappedStorage for MappedStorageEngine {
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
        descriptor_out: &mut gcso_mapped_storage_descriptor_t,
    ) -> GcsoResult<()> {
        if file_path.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *descriptor_out = self.descriptor;
        Ok(())
    }

    #[inline]
    fn close_mmap(&mut self, descriptor: &mut gcso_mapped_storage_descriptor_t) -> GcsoResult<()> {
        descriptor.mapped_address = 0;
        descriptor.fd_handle = -1;
        self.descriptor = *descriptor;
        Ok(())
    }
}

// ===================================================================
// 8.5. 4-Section `.gcso` (mapped-storage) Zero-Copy Container View (`GcsoFourSectionView`)
// ===================================================================

/// Zero-copy view over a 128B-aligned 4-section `.gcso` (mapped-storage) memory-mapped container.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GcsoFourSectionView<'a> {
    /// Section 0a: Validated 128-byte `.gcso` snapshot header (`offset 0..128`).
    pub header: &'a gcso_snapshot_header_t,
    /// Section 0b: Validated 64-byte mapped-storage memory-map & track descriptor (`offset 128..192`).
    pub mapped_storage_descriptor: &'a gcso_mapped_storage_descriptor_t,
    /// Section 1: Zero-copy slice of crystallized macro attractor centroids (`128B` each).
    pub crystals: &'a [CrystallizedAttractorRecord],
    /// Section 2: Zero-copy slice of stigmergic pointer trails (`128B` each).
    pub trails: &'a [gcso_pointer_trail_t],
    /// Section 3a: Optional zero-copy reference to the 64-byte Contrastive Phase Bloom Filter.
    pub bloom: Option<&'a ContrastivePhaseBloomFilter>,
    /// Section 3b: Optional zero-copy reference to the 64-byte FastPathLedger telemetry & shortcut slot.
    pub ledger: Option<&'a gcso_ledger_slot_t>,
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
        has_ledger: bool,
    ) -> GcsoResult<usize> {
        Self::required_byte_len_with_reports(num_crystals, num_trails, has_bloom, has_ledger, 0)
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
        has_ledger: bool,
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
        let Some(off_ledger) = off_bloom.checked_add(bloom_bytes) else {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        };
        let ledger_bytes = if has_ledger {
            size_of::<gcso_ledger_slot_t>()
        } else {
            0
        };
        let Some(off_reports) = off_ledger.checked_add(ledger_bytes) else {
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

    /// Serializes `crystals`, `trails`, `bloom_opt`, `ledger_opt`, and `provenance_reports` into `dst_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < required_byte_len_with_reports(...)`.
    #[inline]
    pub fn write_four_section_with_reports_to_bytes(
        crystals: &[CrystallizedAttractorRecord],
        trails: &[gcso_pointer_trail_t],
        bloom_opt: Option<&ContrastivePhaseBloomFilter>,
        ledger_opt: Option<&gcso_ledger_slot_t>,
        provenance_reports: &[CausalProvenanceReport],
        timestamp_epoch_sec: u64,
        dst_bytes: &mut [u8],
    ) -> GcsoResult<usize> {
        let total_len = Self::required_byte_len_with_reports(
            crystals.len(),
            trails.len(),
            bloom_opt.is_some(),
            ledger_opt.is_some(),
            provenance_reports.len(),
        )?;
        if dst_bytes.len() < total_len {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        let buf = &mut dst_bytes[..total_len];
        let off_crystals = GCSO_FOUR_SECTION_HEADER_BYTES;
        let crystals_bytes = CrystallizedAttractorRecord::write_records_to_container_bytes(
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
            + if ledger_opt.is_some() {
                size_of::<gcso_ledger_slot_t>()
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
        if let Some(ledger) = ledger_opt {
            ledger.write_unaligned_to(buf, off_daes)?;
        }
        CausalProvenanceReport::write_unaligned_slice_to(provenance_reports, buf, off_reports)?;
        if total_len > end_reports {
            buf[end_reports..total_len].fill(0);
        }
        let storage_desc = gcso_mapped_storage_descriptor_t {
            mapped_address: 0,
            file_size_bytes: total_len as u64,
            dma_buffer_handle: 0,
            flags: 4,
            fd_handle: -1,
            reserved: [0; 32],
        };
        storage_desc.write_unaligned_to(buf, GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET)?;

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
        header.ledger_slot_offset = off_daes as u32;
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

    /// Validates and maps a 128-byte aligned 4-section `.gcso` (mapped-storage) container byte slice
    /// into a zero-copy [`GcsoFourSectionView`] ($C_{\mathrm{alloc}} = 0$).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`], or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the container header, alignment, section offsets, or IEEE 802.3 CRC32 checksums are invalid.
    #[inline]
    pub fn from_container_bytes_zero_copy(bytes: &'a [u8]) -> GcsoResult<Self> {
        if bytes.len() < GCSO_FOUR_SECTION_HEADER_BYTES {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        if !is_aligned_to(bytes, 128) {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }

        let header = gcso_snapshot_header_t::from_bytes_at(bytes, 0)?;
        validate_snapshot_header(header, bytes.len())?;
        let mapped_storage_descriptor = gcso_mapped_storage_descriptor_t::from_bytes_at(
            bytes,
            GCSO_MAPPED_STORAGE_PREAMBLE_OFFSET,
        )?;
        if mapped_storage_descriptor.file_size_bytes != 0
            && mapped_storage_descriptor.file_size_bytes != header.total_size
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let total_len = header.total_size as usize;
        let hdr_size = size_of::<gcso_snapshot_header_t>();
        let off_crystals = header.attractor_field_offset as usize;
        let end_crystals = header.dpsr_state_offset as usize;
        let off_trails = header.action_hub_offset as usize;
        let off_bloom = header.edbc_state_offset as usize;
        let off_daes = header.ledger_slot_offset as usize;

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
        let crystals = CrystallizedAttractorRecord::slice_from_container_bytes_zero_copy(
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

        let has_daes = has_daes_flag & ((total_len - off_daes) >= size_of::<gcso_ledger_slot_t>());
        let ledger = if has_daes {
            Some(gcso_ledger_slot_t::from_bytes_at(bytes, off_daes)?)
        } else {
            None
        };
        let off_reports = off_daes
            + if has_daes {
                size_of::<gcso_ledger_slot_t>()
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
}

const _: () = {
    assert!(size_of::<MappedStorageEngine>() == 64);
    assert!(align_of::<MappedStorageEngine>() == 64);
    assert!(offset_of!(MappedStorageEngine, descriptor) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{gcso_ledger_slot_t, gcso_pointer_trail_t, gcso_tagged_ptr_t};
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::{ATTRACTOR_SLOT_PERSONA, MAX_ATTENTION_HEADS};
    use crate::traits::CAUSAL_AXIS_SUBJECTIVE_PERSONA;

    #[test]
    fn test_mapped_storage_engine_dnp_and_pprc_gop_video_stream_seek() {
        let engine = MappedStorageEngine::new(4096, 1);
        assert_eq!(engine.validate(), Ok(()));

        let ledger_slot = gcso_ledger_slot_t {
            mode: 1,
            ..Default::default()
        };
        let mut packet_buf = AlignedBuffer64::new([0u8; 64]);
        let phases = [12i8; MAX_ATTENTION_HEADS];
        let crc = engine
            .encode_dnp_phase_packet(
                &ledger_slot,
                2,
                0x09,
                &phases,
                &mut packet_buf.as_aligned_slice_mut(),
            )
            .unwrap();
        let mut trail = gcso_pointer_trail_t::default();
        let (slot, mode, is_repulsion) = engine
            .decode_and_apply_dnp_packet_to_trail(
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

        let rep_crc = engine
            .encode_dnp_phase_packet(
                &ledger_slot,
                3,
                0x05,
                &phases,
                &mut packet_buf.as_aligned_slice_mut(),
            )
            .unwrap();
        let mut rep_trail = gcso_pointer_trail_t::default();
        let (rep_slot, rep_mode, rep_is_repulsion) = engine
            .decode_and_apply_dnp_packet_to_trail(
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
            .pack_kv_gop_stream_bytes(&iframe, &gop_pframes, &mut stream_bytes)
            .unwrap();
        assert_eq!(written, 256);

        let mut seek_trail = gcso_pointer_trail_t::default();
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
        assert_eq!(gcso_tagged_ptr_t(seek_trail.current_ptr).unpack_addr(), 102);
        assert!(lp > -0.05 && (r_bar - 1.0).abs() < 1.0e-4 && kappa > 10.0 && v_shift > 0.0);
    }
}
