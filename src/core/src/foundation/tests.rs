// name: src/core/src/foundation/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::liminal::{
    coupled_slot_count, coupled_slot_row_mask, is_slot_pair_coupled, pack_action_slot_tag,
    unpack_action_mode_from_tag, unpack_attractor_slot_from_tag, with_slot_pair_coupled_symmetric,
    ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
};

#[test]
fn test_action_slot_tag_and_coupling_matrix_primitives() {
    let tag = pack_action_slot_tag(ATTRACTOR_SLOT_PERSONA, ACTION_PRESET_FULL_ORCHESTRATION);
    assert_eq!(unpack_attractor_slot_from_tag(tag), ATTRACTOR_SLOT_PERSONA);
    assert_eq!(
        unpack_action_mode_from_tag(tag),
        ACTION_PRESET_FULL_ORCHESTRATION
    );

    let mut matrix = 0u64;
    assert!(!is_slot_pair_coupled(
        matrix,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE)
    ));

    matrix = with_slot_pair_coupled_symmetric(
        matrix,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
        true,
    );
    assert!(is_slot_pair_coupled(
        matrix,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE)
    ));
    assert_eq!(
        coupled_slot_row_mask(matrix, u32::from(ATTRACTOR_SLOT_PERSONA)),
        1 << ATTRACTOR_SLOT_KNOWLEDGE
    );
    assert_eq!(
        coupled_slot_count(matrix, u32::from(ATTRACTOR_SLOT_PERSONA)),
        1
    );

    matrix = with_slot_pair_coupled_symmetric(
        matrix,
        u32::from(ATTRACTOR_SLOT_PERSONA),
        u32::from(ATTRACTOR_SLOT_KNOWLEDGE),
        false,
    );
    assert_eq!(matrix, 0);
}

#[test]
fn test_pod_tagged_ptr_and_incremental_crc32() {
    let ptr = gcso_tagged_ptr_t::pack(0x1234_5678, 0x05, 16);
    assert_eq!(ptr.unpack_addr(), 0x1234_5678);
    assert_eq!(ptr.unpack_tag(), 0x05);
    assert_eq!(ptr.unpack_phase(), 16);

    let full_crc = compute_crc32_ieee(b"123456789");
    assert_eq!(full_crc, 0xCBF4_3926);
    let part1 = update_crc32_ieee(0, b"1234");
    let part2 = update_crc32_ieee(part1, b"56789");
    assert_eq!(part2, full_crc);
}
