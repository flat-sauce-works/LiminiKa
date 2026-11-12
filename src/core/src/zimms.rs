// name: src/core/src/zimms.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete 64-byte cacheline-aligned [`ZimmsEngine`] `Pod` state descriptor applying the
//! [`PprcCache`], [`DnpPacketDispatcher`], and [`ZimmsStorage`] traits from [`crate::traits`].

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]

use crate::abi::{
    gcso_config_t, gcso_pprc_keyframe_header_t, gcso_snapshot_header_t, gcso_zimms_descriptor_t,
    GCSO_ERROR_BUFFER_TOO_SMALL, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::primitives::{init_snapshot_header, verify_snapshot_payload_crc32};
use crate::traits::{DnpPacketDispatcher, GcsoResult, Pod, PprcCache, ZimmsStorage};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Zero-Copy In-Memory Mapped Storage (ZIMMS `.gcso` containers), PPRC keyframe caching,
/// and DNP 64-byte phase packet dispatch.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZimmsEngine {
    /// Embedded 64-byte C-ABI v0.1.1 ZIMMS memory-map & track descriptor (`offset 0..64`).
    pub descriptor: gcso_zimms_descriptor_t,
}

// SAFETY: `ZimmsEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `gcso_zimms_descriptor_t`.
unsafe impl Pod for ZimmsEngine {}

impl Default for ZimmsEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(0, 1)
    }
}

impl ZimmsEngine {
    /// Constructs a canonical 64-byte [`ZimmsEngine`] initialized with `file_size_bytes` and `flags`.
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

impl PprcCache for ZimmsEngine {
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

impl DnpPacketDispatcher for ZimmsEngine {}

impl ZimmsStorage for ZimmsEngine {
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

const _: () = {
    assert!(size_of::<ZimmsEngine>() == 64);
    assert!(align_of::<ZimmsEngine>() == 64);
    assert!(offset_of!(ZimmsEngine, descriptor) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{gcso_daes_slot_t, gcso_pointer_trail_t};
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::MAX_ATTENTION_HEADS;

    #[test]
    fn test_zimms_engine_and_dnp_round_trip() {
        let engine = ZimmsEngine::new(4096, 1);
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
    }
}
