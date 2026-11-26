// name: src/core/src/runtime/storage/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::foundation::{AlignedBuffer64, AlignedBuffer128, TaggedPtr, compute_crc32_ieee};
use crate::liminal::{
    LMNK_ERR_ALIGNMENT, LMNK_ERR_BUFFER_TOO_SMALL, LMNK_ERR_CONTAINER_CORRUPTED,
    LMNK_ERR_TOPOLOGY_MISMATCH, LMNK_ERR_VERSION_MISMATCH, MAX_ATTENTION_HEADS,
};
use crate::runtime::gateway::{CAUSAL_AXIS_SUBJECTIVE_PERSONA, CausalProvenanceReport};
use crate::runtime::topology::CORE_TOPOLOGY;
use crate::stigmergy::{
    ATTRACTOR_SLOT_PERSONA, ContrastivePhaseBloomFilter, LedgerSlot, PhaseAnchorRecord,
    PointerTrail,
};

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
        StandardSectionsInput {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            anchors: &anchors,
            trails: &trails,
            bloom: Some(&bloom),
            ledger: Some(&ledger),
            provenance_reports: &reports,
            timestamp_epoch_sec: 1_774_900_000,
        },
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
    // while section_bytes() and read_phase_anchor() succeed cleanly.
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
