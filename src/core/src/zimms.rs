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
    gcso_config_t, gcso_pprc_keyframe_header_t, gcso_snapshot_header_t, gcso_zimms_descriptor_t,
    GCSO_ERROR_BUFFER_TOO_SMALL, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::primitives::{init_snapshot_header, verify_snapshot_payload_crc32};
use crate::traits::{DnpPacketDispatcher, GcsoResult, Pod, PprcCache, ZimmsStorage};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Zero-Copy In-Memory Mapped Storage (ZIMMS `.gcso` containers), PPRC keyframe/delta video-stream
/// memory & persona reproduction, and DNP 64-byte phase packet dispatch.
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
    use crate::abi::{gcso_daes_slot_t, gcso_pointer_trail_t, gcso_tagged_ptr_t};
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::{ATTRACTOR_SLOT_PERSONA, MAX_ATTENTION_HEADS};
    use crate::traits::CAUSAL_AXIS_SUBJECTIVE_PERSONA;

    #[test]
    fn test_zimms_engine_dnp_and_pprc_gop_video_stream_seek() {
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
