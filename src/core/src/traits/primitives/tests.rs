// name: src/core/src/traits/primitives/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#[allow(unused_imports)]
use super::{constants_pod::*, lut_crc32::*, packets_ffi::*, scalar_math::*, swar_ops::*};
use super::*;

#[test]
fn test_opaque_handles_type_safety() {
    assert_eq!(size_of::<GcsoContextHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoContainerHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoActionHubHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoLedgerSlotHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoAttractorFieldHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoEdbcControllerHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoDpsrKernelHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoSubHeadRouterHandle>(), size_of::<*mut ()>());
    assert_eq!(size_of::<GcsoRank1ResidualHandle>(), size_of::<*mut ()>());
}

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
fn test_ffi_handle_ref_mut_cast_and_disjoint_validation() {
    let mut config = gcso_config_t::default();
    let raw_ptr: *mut gcso_config_t = &mut config;

    // SAFETY: `raw_ptr` points to a valid stack-allocated `gcso_config_t`.
    let handle_ref = unsafe { FfiHandleRef::from_raw(raw_ptr) }.unwrap();
    let cloned_ref = handle_ref;
    assert_eq!(cloned_ref.read_copy().head_dim, 0);

    let cast_u32 = handle_ref.try_cast::<u32>().unwrap();
    assert_eq!(*cast_u32, 0);

    // SAFETY: `raw_ptr` is uniquely borrowed here.
    let mut handle_mut = unsafe { FfiHandleMut::from_raw(raw_ptr) }.unwrap();
    handle_mut.head_dim = 128;
    assert_eq!(handle_mut.into_mut().head_dim, 128);
    assert_eq!(config.head_dim, 128);

    let mut a = 1u32;
    let mut b = 2u32;
    let mut c = 3u32;
    // SAFETY: `a`, `b`, and `c` are distinct stack variables.
    unsafe {
        let (mut ha, mut hb, mut hc) =
            FfiHandleMut::from_raw_disjoint_triple(&mut a, &mut b, &mut c).unwrap();
        ha.write(10);
        hb.write(20);
        hc.write(30);
        assert!(FfiHandleMut::from_raw_disjoint_pair(&mut a, &mut a).is_err());
    }
    assert_eq!((a, b, c), (10, 20, 30));

    let null_ptr: *const gcso_config_t = core::ptr::null();
    // SAFETY: Testing null pointer rejection and optional null handling.
    unsafe {
        assert_eq!(
            FfiHandleRef::from_raw(null_ptr),
            Err(GCSO_ERROR_NULL_POINTER)
        );
        assert!(FfiHandleRef::from_raw_opt(null_ptr).unwrap().is_none());
    }
}

#[test]
fn test_pod_unaligned_bulk_io_and_incremental_crc32() {
    let hdr = init_descriptor_header::<gcso_config_t>();
    assert_eq!(validate_descriptor_header::<gcso_config_t>(&hdr), Ok(()));

    let mut buf = [0u8; 32];
    let src_arr = [hdr, hdr];
    assert_eq!(
        gcso_descriptor_header_t::write_unaligned_slice_to(&src_arr, &mut buf, 3),
        Ok(())
    );
    let mut dst_arr = [gcso_descriptor_header_t::default(); 2];
    assert_eq!(
        gcso_descriptor_header_t::read_unaligned_slice_from(&buf, 3, &mut dst_arr),
        Ok(())
    );
    assert_eq!(dst_arr, src_arr);

    let full_crc = compute_crc32_ieee(b"123456789");
    assert_eq!(full_crc, 0xCBF4_3926);
    let part1 = update_crc32_ieee(0, b"1234");
    let part2 = update_crc32_ieee(part1, b"56789");
    assert_eq!(part2, full_crc);

    let mut container = [0u8; 256];
    container[128..137].copy_from_slice(b"123456789");
    let mut snap = init_snapshot_header(256, 1_774_900_000);
    let stamped_crc = finalize_snapshot_header_crc32(&mut snap, &container).unwrap();
    assert_eq!(stamped_crc, snap.checksum_crc32);
    assert_eq!(verify_snapshot_payload_crc32(&snap, &container), Ok(()));

    container[128] ^= 0xFF;
    assert_eq!(
        verify_snapshot_payload_crc32(&snap, &container),
        Err(GCSO_ERROR_CONTAINER_CORRUPTED)
    );
}
