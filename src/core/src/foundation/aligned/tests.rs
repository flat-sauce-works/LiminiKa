// name: src/core/src/foundation/aligned/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for aligned slices, disjoint guards, and stack buffers.

use super::*;
use crate::liminal::{MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PhaseQ7};

#[test]
fn test_infallible_aligned_buffers_range_indexing_and_upgrade() {
    let mut buf64 = AlignedBuffer64::<f32, 64>::splat(1.5);
    assert_eq!(&buf64[0..4], &[1.5f32; 4]);
    buf64[2..4].copy_from_slice(&[4.0, 5.0]);
    assert_eq!(buf64[2], 4.0);

    let slice = buf64.as_aligned_slice();
    assert_eq!(slice.align(), 64);
    assert_eq!(slice.len(), 64);
    assert_eq!(slice.byte_len(), 256);

    let mut slice_mut = buf64.as_aligned_slice_mut();
    slice_mut[0] = 3.0;
    let relaxed: AlignedSliceMut16<'_, f32> = slice_mut.relax_alignment::<16>();
    assert_eq!(relaxed[0], 3.0);

    let zero_buf = AlignedBuffer32::<f32, 16>::zeroed();
    assert_eq!(zero_buf[0], 0.0);
    let arr_buf = AlignedBuffer32::<f32, 16>::new([2.5f32; 16]);
    assert_eq!(arr_buf[15], 2.5);
}

#[test]
fn test_disjoint_mut_pair_triple_and_ref_mut() {
    let mut buf_q = AlignedBuffer32::<f32, 32>::splat(1.0);
    let mut buf_k = AlignedBuffer32::<f32, 32>::splat(2.0);
    let mut buf_v = AlignedBuffer32::<f32, 32>::splat(3.0);

    let mut triple = DisjointMutTriple::new(
        buf_q.as_aligned_slice_mut(),
        buf_k.as_aligned_slice_mut(),
        buf_v.as_aligned_slice_mut(),
    )
    .unwrap();

    {
        let mut reborrowed = triple.reborrow();
        reborrowed.first[0] += reborrowed.second[0] + reborrowed.third[0];
    }
    assert_eq!(buf_q[0], 6.0);

    let mut ref_mut =
        DisjointRefMut::new(buf_k.as_aligned_slice(), buf_q.as_aligned_slice_mut()).unwrap();
    let (inp, out) = ref_mut.as_ref_mut();
    out[1] = inp[1] * 4.0;
    assert_eq!(buf_q[1], 8.0);
}

#[test]
fn test_cast_and_split_at() {
    let mut buf = AlignedBuffer64::<f32, 32>::splat(1.0);
    let mut slice_mut = buf.as_aligned_slice_mut();
    let (mut left, mut right) = slice_mut.split_at_mut(16).unwrap();
    left[0] = 10.0;
    right[0] = 20.0;
    assert_eq!(buf[0], 10.0);
    assert_eq!(buf[16], 20.0);

    let slice_u8: AlignedSlice64<'_, u8> = buf.as_aligned_slice().try_cast::<u8>().unwrap();
    assert_eq!(slice_u8.len(), 128);
}

#[test]
fn test_attractor_slot_bank_triple_and_micro_swarm_chunks() {
    let mut multi_slot_bank =
        AlignedBuffer64::<PhaseQ7, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();

    {
        let bank_2d_mut = multi_slot_bank
            .as_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
            .unwrap();
        bank_2d_mut[0][0] = 12;
    }
    assert_eq!(multi_slot_bank[0], 12);

    {
        let k_slot = multi_slot_bank
            .slot_exact_mut::<MAX_ATTENTION_HEADS>(2)
            .unwrap();
        k_slot.fill(56);
    }

    let bank_2d = multi_slot_bank
        .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()
        .unwrap();
    assert_eq!(bank_2d[0][0], 12);
    assert_eq!(bank_2d[2][0], 56);

    let imm_slice = multi_slot_bank.as_aligned_slice();
    let k_ref = imm_slice.slot_exact::<MAX_ATTENTION_HEADS>(2).unwrap();
    assert_eq!(k_ref[0], 56);
}
