// name: src/core/src/foundation/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use core::f32::consts::PI;

use super::*;
use crate::foundation::lut_crc32::{crc32_step_block64_words, CRC32_UNIFORM56_PAD8_ZERO_SEED_REF};
use crate::foundation::rope::visual::rotate_so2_3d_txy_slice_simd;
use crate::foundation::simd_stats::{
    norm_l2_underflow_resilient_f32, rotate_so2_slice_unrolled16, scale_f32_slice_unrolled8,
    sincos_f32_x8_simd,
};
use crate::foundation::swar_simd::{
    eval_fused_phase_clamp_subhead_sincos_q7_fast, eval_phase_clamp_sincos_q7_fast,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, MAX_ATTENTION_HEADS, Q7_PHASE_SCALE_RAD,
};

#[test]
fn test_action_slot_tag_primitives() {
    let tag = pack_action_slot_tag(1, ACTION_PRESET_FULL_ORCHESTRATION);
    assert_eq!(unpack_attractor_slot_from_tag(tag), 1);
    assert_eq!(
        unpack_action_mode_from_tag(tag),
        ACTION_PRESET_FULL_ORCHESTRATION
    );
}

#[test]
fn test_pod_tagged_ptr_and_incremental_crc32() {
    let ptr = TaggedPtr::pack(0x1234_5678, 0x05, 16);
    assert_eq!(ptr.unpack_addr(), 0x1234_5678);
    assert_eq!(ptr.unpack_tag(), 0x05);
    assert_eq!(ptr.unpack_phase(), 16);

    let full_crc = compute_crc32_ieee(b"123456789");
    assert_eq!(full_crc, 0xCBF4_3926);
    let part1 = update_crc32_ieee(0, b"1234");
    let part2 = update_crc32_ieee(part1, b"56789");
    assert_eq!(part2, full_crc);
}

#[test]
fn test_branchless_inv_pos_crc32_and_constant_time_swar() {
    assert_eq!(branchless_select_u32(true, 10, 20), 10);
    assert_eq!(branchless_select_f32(false, 1.5, -2.5), -2.5);
    assert!((branchless_inv_pos_f32(4.0) - 0.25).abs() < 1.0e-6);
    assert_eq!(branchless_inv_pos_f32(0.0), 0.0);
    assert_eq!(branchless_inv_pos_f32(-2.0), 0.0);
    assert_eq!(branchless_inv_pos_f32(f32::INFINITY), 0.0);

    // Standard IEEE 802.3 CRC32 test vector ("123456789" -> 0xCBF43926)
    assert_eq!(branchless_crc32_ieee(b"123456789"), 0xCBF4_3926);
    let chained = branchless_update_crc32_ieee(branchless_crc32_ieee(b"1234"), b"56789");
    assert_eq!(chained, 0xCBF4_3926);
    for b_val in [0u8, 1, 28, 127, 128, 255] {
        let unif_bytes = [b_val; 128];
        let w_unif = [u64::from_le_bytes([b_val; 8]); 16];
        assert_eq!(
            branchless_crc32_128b_words(&w_unif),
            branchless_crc32_ieee(&unif_bytes)
        );
    }
    let long_payload = [0x5Au8; 512];
    let crc_full = branchless_crc32_ieee(&long_payload);
    let crc_split = branchless_update_crc32_ieee(
        branchless_crc32_ieee(&long_payload[..256]),
        &long_payload[256..],
    );
    assert_eq!(crc_full, crc_split);

    let a = swar_q7_pack_x8(&[100, -100, 64, -64, 127, -128, 0, 12]);
    let b = swar_q7_pack_x8(&[50, -50, -64, 64, 10, -10, 0, -12]);
    let sum = swar_q7_unpack_x8(swar_q7_add_sat_x8(a, b));
    assert_eq!(sum, [127, -128, 0, 0, 127, -128, 0, 0]);

    let w_a = [swar_q7_pack_x8(&[-120; 8]); 8];
    let w_b = [swar_q7_pack_x8(&[120; 8]); 8];
    let bias_64 = 64u64.wrapping_mul(SWAR_SPLAT_BYTE_64);
    let (slerp_words, is_uni) =
        swar_slerp_words_with_uniform_flag_64(&w_a, true, &w_b, true, 64, bias_64);
    assert!(is_uni);
    for lane in swar_q7_unpack_x8(slerp_words[0]) {
        assert!(lane.unsigned_abs() >= 120);
    }

    for deg in -180..=180 {
        let rad = (deg as f32) * (PI / 180.0);
        let (s, c) = branchless_sincos_f32(rad);
        let q7_ref = branchless_rad_to_q7(branchless_atan2_f32(s, c));
        let q7_fast = branchless_atan2_to_q7_fast(s, c);
        assert!((i32::from(q7_ref) - i32::from(q7_fast)).abs() <= 1);
    }
}

#[test]
fn test_unified_manifold_fold_gravity_fold_toroidal_centroid_step_and_limina_log_prob() {
    let trail = [32i8; MAX_ATTENTION_HEADS];
    let anchors = [
        [-96i8; MAX_ATTENTION_HEADS],
        [32i8; MAX_ATTENTION_HEADS],
        [28i8; MAX_ATTENTION_HEADS],
    ];
    let energies = [0.5_f32, 3.0_f32, 2.0_f32];
    let mut folded = [0i8; MAX_ATTENTION_HEADS];

    let (tot_pull, mean_coh, dom_idx, act_cnt) =
        branchless_q7_unified_manifold_fold_64(&trail, &anchors, &energies, &mut folded);
    assert_eq!(dom_idx, 1);
    assert_eq!(act_cnt, 3);
    assert!(tot_pull > 8.0 && mean_coh > 0.70);
    assert!((i32::from(folded[0]) - 30).abs() <= 4);

    let mut grav_folded = [0i8; MAX_ATTENTION_HEADS];
    let (grav_pull, grav_coh, grav_dom, grav_cnt) =
        branchless_q7_gravity_fold_64(&trail, &anchors, &energies, 0.25, &mut grav_folded);
    assert_eq!(grav_dom, 1);
    assert_eq!(grav_cnt, 3);
    assert!(grav_pull > 15.0 && grav_coh > 0.85);

    let (coh_ab, r_bar, mean_q7, kappa) =
        branchless_q7_coherence_and_resultant_64(&trail, &anchors[1]);
    assert!((coh_ab - 1.0).abs() < 1.0e-4);
    let mut nonunif_a = [0i8; MAX_ATTENTION_HEADS];
    let mut nonunif_b = [71i8; MAX_ATTENTION_HEADS];
    nonunif_a[0] = 1;
    nonunif_b[0] = 72;
    let coh_71 = branchless_q7_coherence_64(&nonunif_a, &nonunif_b);
    assert!((coh_71 - (-0.170_962)).abs() < 2.0e-3);
    assert!((r_bar - 1.0).abs() < 1.0e-4);
    assert_eq!(mean_q7, 32);
    assert!(kappa > 10.0);

    let w_init = load_q7_words_64(&trail);
    let (w_step1, e_step1, rb_step1, kp_step1) =
        branchless_toroidal_centroid_step_words_64(&w_init, 0, 0.0, 0.0, 0.0, &w_init, 2.5);
    assert_eq!(w_step1, w_init);
    assert!((e_step1 - 2.5).abs() < 1.0e-5 && (rb_step1 - 1.0).abs() < 1.0e-4 && kp_step1 > 10.0);
    let (w_stat, e_stat, rb_stat, kp_stat) = branchless_toroidal_centroid_step_words_64(
        &w_step1, 1, e_step1, rb_step1, kp_step1, &w_init, 2.0,
    );
    assert_eq!(w_stat, w_init);
    assert!((e_stat - 3.5).abs() < 1.0e-5 && rb_stat == rb_step1 && kp_stat == kp_step1);

    let (coh_same, lp_same) = branchless_q7_transition_log_prob_64(&trail, &anchors[1]);
    assert!((coh_same - 1.0).abs() < 1.0e-4 && lp_same.abs() < 1.0e-4);

    let emb = [0.5_f32; 128];
    let proj = branchless_project_f32_slice_to_q7_64(&emb, 1.0);
    assert_eq!(proj[0], proj[63]);
    assert!(proj[0] > 15);
}

#[test]
fn test_avx2_sincos_x8_3d_txy_lockstep_qkv_and_subnormal_norm_rescue() {
    let angles = [-3.0_f32, -1.5, -0.5, 0.0, 0.5, 1.2, 2.4, 3.1];
    let mut sin_out = [0.0_f32; 8];
    let mut cos_out = [0.0_f32; 8];
    sincos_f32_x8_simd(&angles, &mut sin_out, &mut cos_out);
    for i in 0..8 {
        let (s_ref, c_ref) = branchless_sincos_f32(angles[i]);
        assert!((sin_out[i] - s_ref).abs() < 1.0e-4);
        assert!((cos_out[i] - c_ref).abs() < 1.0e-4);
    }

    let mut grid_3d = [1.0_f32; 48];
    let (st, ct) = branchless_sincos_q7(16);
    let (sx, cx) = branchless_sincos_q7(32);
    let (sy, cy) = branchless_sincos_q7(-16);
    rotate_so2_3d_txy_slice_simd(&mut grid_3d, st, ct, sx, cx, sy, cy);
    for hex in grid_3d.chunks_exact(6) {
        assert!(((hex[0] * hex[0] + hex[1] * hex[1]) - 2.0).abs() < 1.0e-5);
        assert!(((hex[2] * hex[2] + hex[3] * hex[3]) - 2.0).abs() < 1.0e-5);
        assert!(((hex[4] * hex[4] + hex[5] * hex[5]) - 2.0).abs() < 1.0e-5);
    }

    let mut q = [1.0_f32; 64];
    let mut k = [1.0_f32; 64];
    let mut v = [1.0_f32; 64];
    rotate_so2_slice_unrolled16(&mut q, st, ct);
    rotate_so2_slice_unrolled16(&mut k, st, ct);
    rotate_so2_slice_unrolled16(&mut v, st, ct);
    assert!((q[0] - k[0]).abs() < 1.0e-6 && (q[0] - v[0]).abs() < 1.0e-6);

    // Verify constant-time identity rotation preserves exact values
    let mut q_id = [1.5_f32; 64];
    rotate_so2_slice_unrolled16(&mut q_id, 0.0, 1.0);
    assert_eq!(q_id, [1.5_f32; 64]);

    // Verify AVX2 inline subnormal flushing in `scale_f32_slice_unrolled8`
    let mut sub_buf = [2.0e-38_f32; 64];
    scale_f32_slice_unrolled8(&mut sub_buf, 0.5);
    assert_eq!(sub_buf[0], 0.0);

    let tiny = [1.0e-22_f32; 64];
    let (norm, ok) = norm_l2_underflow_resilient_f32(&tiny);
    assert!(ok && (norm - 8.0e-22).abs() < 1.0e-26);

    let sample_vec = [-2.0_f32, 1.0, 3.0, -4.0];
    let (s_abs, s_sq, m_abs) = sum_abs_sq_and_max_f32_simd8(&sample_vec);
    assert!((s_abs - 10.0).abs() < 1.0e-5);
    assert!((s_sq - 30.0).abs() < 1.0e-5);
    assert!((m_abs - 4.0).abs() < 1.0e-5);

    // Verify single-evaluation fused `PhaseClamp` + sub-head router angle helper
    let scale_018 = Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(0.18);
    let (s1, c1) = eval_phase_clamp_sincos_q7_fast(28, false, 0.18, scale_018);
    let (s2, c2) = eval_phase_clamp_sincos_q7_fast(12, false, 0.18, scale_018);
    let (sf, cf) = eval_fused_phase_clamp_subhead_sincos_q7_fast(
        28, false, false, 0.18, scale_018, 12, false, false, 0.18, scale_018,
    );
    assert!((sf - (s1 * c2 + c1 * s2)).abs() < 1.0e-5);
    assert!((cf - (c1 * c2 - s1 * s2)).abs() < 1.0e-5);

    // Verify 1KB GF(2) CRC32_UNIFORM56_PAD8_ZERO_SEED_REF
    let w_trail_blk1 = [
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0x1818_1818_1818_1818u64,
        0u64,
    ];
    let mut raw_blk1 = [0x18u8; 64];
    raw_blk1[56..64].fill(0);
    assert_eq!(
        crc32_block64_zero_seed_words(&w_trail_blk1),
        CRC32_UNIFORM56_PAD8_ZERO_SEED_REF[0x18]
    );
    assert_eq!(
        crc32_step_block64_words(0xFFFF_FFFFu32, &w_trail_blk1) ^ 0xFFFF_FFFFu32,
        branchless_crc32_ieee(&raw_blk1)
    );
}

#[test]
fn test_branchless_mask8_min_delta_and_argmin_f32_x8_masked() {
    // Empty universe -> (0, 0, true) on tie
    assert_eq!(branchless_mask8_min_delta(0x00, 0xFF), (0x00, 0, true));
    assert_eq!(branchless_mask8_min_delta(0x00, 0x00), (0x00, 0, true));

    // Tie-breaking: inside wins when popcount(inside) == popcount(outside)
    assert_eq!(branchless_mask8_min_delta(0xFF, 0x0F), (0x0F, 4, true));
    assert_eq!(branchless_mask8_min_delta(0x33, 0x0F), (0x03, 2, true));

    // Inside strictly smaller
    assert_eq!(branchless_mask8_min_delta(0xFF, 0x05), (0x05, 2, true));
    // Outside strictly smaller
    assert_eq!(branchless_mask8_min_delta(0xFF, 0xFE), (0x01, 1, false));
    // Single-bit universe
    assert_eq!(branchless_mask8_min_delta(0x10, 0x10), (0x00, 0, false));
    assert_eq!(branchless_mask8_min_delta(0x10, 0x00), (0x00, 0, true));

    // Argmin: empty eligible mask -> (0xFF, +inf)
    let vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (lane, val) = branchless_argmin_f32_x8_masked(&vals, 0x00);
    assert_eq!(lane, 0xFF);
    assert!(val.is_infinite() && val.is_sign_positive());

    // Argmin: all NaN -> (0xFF, +inf)
    let all_nan = [f32::NAN; 8];
    let (lane, val) = branchless_argmin_f32_x8_masked(&all_nan, 0xFF);
    assert_eq!(lane, 0xFF);
    assert!(val.is_infinite() && val.is_sign_positive());

    // Argmin: all +inf -> must select lowest eligible lane, NOT 0xFF
    let all_inf = [f32::INFINITY; 8];
    let (lane, val) = branchless_argmin_f32_x8_masked(&all_inf, 0b0101_0000);
    assert_eq!(lane, 4);
    assert!(val.is_infinite() && val.is_sign_positive());

    // Argmin: single-bit mask
    let (lane, val) = branchless_argmin_f32_x8_masked(&vals, 1 << 5);
    assert_eq!(lane, 5);
    assert_eq!(val, 6.0);

    // Argmin: tie-breaking chooses lowest eligible index
    let tied = [3.0, 1.5, 1.5, 2.0, 1.5, 4.0, 0.5, 0.5];
    assert_eq!(
        branchless_argmin_f32_x8_masked(&tied, 0b0011_1111),
        (1, 1.5)
    );
    assert_eq!(branchless_argmin_f32_x8_masked(&tied, 0xFF), (6, 0.5));

    // Argmin: negative values and -inf
    let negs = [-1.0, -5.5, -2.0, -5.5, f32::NEG_INFINITY, 0.0, 1.0, -10.0];
    assert_eq!(
        branchless_argmin_f32_x8_masked(&negs, 0b0000_1111),
        (1, -5.5)
    );
    assert_eq!(
        branchless_argmin_f32_x8_masked(&negs, 0xFF),
        (4, f32::NEG_INFINITY)
    );

    // Argmin: mixed NaN resilience
    let mixed = [
        f32::NAN,
        f32::INFINITY,
        -0.25,
        f32::NAN,
        -0.25,
        2.0,
        f32::NAN,
        0.0,
    ];
    assert_eq!(
        branchless_argmin_f32_x8_masked(&mixed, 0b0000_0011),
        (1, f32::INFINITY)
    );
    assert_eq!(branchless_argmin_f32_x8_masked(&mixed, 0xFF), (2, -0.25));
}

#[test]
fn test_branchless_argmin_u8_x8_masked() {
    let vals = [10u8, 20, 30, 40, 50, 60, 70, 80];
    // Empty eligible mask -> (0xFF, u8::MAX)
    assert_eq!(branchless_argmin_u8_x8_masked(&vals, 0x00), (0xFF, u8::MAX));

    // Single eligible lane
    assert_eq!(branchless_argmin_u8_x8_masked(&vals, 1 << 5), (5, 60));

    // All lanes eligible
    assert_eq!(branchless_argmin_u8_x8_masked(&vals, 0xFF), (0, 10));

    // Extreme bounds 0 and 255 (all u8::MAX must pick lowest eligible lane, not 0xFF)
    let all_max = [u8::MAX; 8];
    assert_eq!(
        branchless_argmin_u8_x8_masked(&all_max, 0b0101_0000),
        (4, u8::MAX)
    );
    let with_zero = [255u8, 128, 0, 0, 255, 1, 0, 255];
    assert_eq!(branchless_argmin_u8_x8_masked(&with_zero, 0xFF), (2, 0));
    assert_eq!(
        branchless_argmin_u8_x8_masked(&with_zero, 0b1111_0000),
        (6, 0)
    );

    // Lowest-index tie-breaking
    let tied = [40u8, 15, 15, 20, 15, 99, 5, 5];
    assert_eq!(branchless_argmin_u8_x8_masked(&tied, 0b0011_1111), (1, 15));
    assert_eq!(branchless_argmin_u8_x8_masked(&tied, 0xFF), (6, 5));
}

#[test]
fn test_branchless_q7_weighted_ensemble_words_64() {
    use crate::liminal::{PhaseQ7, NUM_STIGMERGIC_GRANULES};
    let base_zero = [0u64; NUM_STIGMERGIC_GRANULES];
    let p0 = [64i8; MAX_ATTENTION_HEADS];
    let p1 = [-64i8; MAX_ATTENTION_HEADS];
    let p2 = [120i8; MAX_ATTENTION_HEADS];
    let p3 = [-120i8; MAX_ATTENTION_HEADS];
    let p_zero = [0i8; MAX_ATTENTION_HEADS];
    let profiles: [&[PhaseQ7; MAX_ATTENTION_HEADS]; 8] =
        [&p0, &p1, &p2, &p3, &p_zero, &p_zero, &p_zero, &p_zero];
    let weights = [128u8, 64, 96, 96, 0, 0, 0, 0];

    // 1. Empty mask returns base unchanged
    let (w_empty, d_empty, entry_empty) =
        branchless_q7_weighted_ensemble_words_64(profiles, &weights, 0, &base_zero);
    assert_eq!(w_empty, base_zero);
    assert_eq!(d_empty, 0);
    assert_eq!(entry_empty, 0);

    // 2. Single lane: weight=128 reproduces exact profile, weight=96 scales by 3/4
    let (w_full, d_full, entry_full) =
        branchless_q7_weighted_ensemble_words_64(profiles, &weights, 0b0000_0001, &base_zero);
    assert_eq!(entry_full, 0);
    assert_eq!(d_full, 127);
    assert_eq!(swar_q7_unpack_x8(w_full[0])[0], 64);

    let (w_single, d_single, entry_single) =
        branchless_q7_weighted_ensemble_words_64(profiles, &weights, 0b0000_0100, &base_zero);
    assert_eq!(entry_single, 2);
    assert_eq!(d_single, 96);
    assert_eq!(swar_q7_unpack_x8(w_single[0])[0], 90);

    // 3. Multi-lane symmetric rounding (positive and negative head contributions)
    let mut mixed_pos = [0i8; MAX_ATTENTION_HEADS];
    mixed_pos[0] = 65;
    mixed_pos[1] = -65;
    let profiles_mix: [&[PhaseQ7; MAX_ATTENTION_HEADS]; 8] = [
        &mixed_pos, &mixed_pos, &mixed_pos, &p_zero, &p_zero, &p_zero, &p_zero, &p_zero,
    ];
    let w_mix = [128u8; 8];
    let (w_3, d_3, entry_3) =
        branchless_q7_weighted_ensemble_words_64(profiles_mix, &w_mix, 0b0000_0111, &base_zero);
    assert_eq!(entry_3, 0);
    assert_eq!(d_3, 127);
    let lanes_3 = swar_q7_unpack_x8(w_3[0]);
    // 3 * 65 * 43 = 8385 -> (8385 + 64) / 128 = 66; negative -> -66
    assert_eq!(lanes_3[0], 66);
    assert_eq!(lanes_3[1], -66);

    // 4. Circular SWAR wraparound onto non-zero base
    let base_100 = load_q7_words_64(&[100i8; MAX_ATTENTION_HEADS]);
    let (w_wrap, _, _) =
        branchless_q7_weighted_ensemble_words_64(profiles, &weights, 0b0000_0001, &base_100);
    // 100 + 64 = 164 wraps on S^1 Q7 to -92
    assert_eq!(swar_q7_unpack_x8(w_wrap[0])[0], -92);

    // 5. Contiguous-matrix variant produces bit-exact output without stack pointer array
    let contig = [p0, p1, p2, p3, p_zero, p_zero, p_zero, p_zero];
    for mask in [0u8, 0b0000_0001, 0b0000_0100, 0b0000_1111] {
        assert_eq!(
            branchless_q7_weighted_ensemble_contiguous_words_64(&contig, &weights, mask, &base_100),
            branchless_q7_weighted_ensemble_words_64(profiles, &weights, mask, &base_100)
        );
    }
}
