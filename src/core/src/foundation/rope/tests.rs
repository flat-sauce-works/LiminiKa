// name: src/core/src/foundation/rope/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::foundation::aligned::{AlignedBuffer16, AlignedBuffer32, AlignedBuffer64};
use crate::foundation::pod::{gcso_paged_bitmask_t, gcso_subhead_config_t};
use crate::foundation::scalar_math::DEFAULT_PHASE_CLAMP_MAX_RAD;
use crate::liminal::gcso_q7_t;

struct DummyRopeEngine {
    head_dim: u32,
    num_heads: u32,
}

impl PhaseSteering for DummyRopeEngine {
    fn head_dim(&self) -> u32 {
        self.head_dim
    }
    fn num_heads(&self) -> u32 {
        self.num_heads
    }
}

#[test]
fn test_phase_steering_procrustes_damping_and_autonomous_aperture() {
    let eng = DummyRopeEngine {
        head_dim: 8,
        num_heads: 8,
    };
    let mut src_t = AlignedBuffer32::new([0.0_f32; 64]);
    for pair in src_t.data.chunks_exact_mut(2) {
        pair[0] = 1.0;
        pair[1] = 0.0;
    }
    let mut tgt_t = src_t;
    let p32 = AlignedBuffer16::new([32i8; 64]);
    eng.apply_phase_steering(&mut tgt_t.as_aligned_slice_mut(), &p32.as_aligned_slice())
        .unwrap();
    let mut proc_out = AlignedBuffer16::new([0i8; 64]);
    eng.compute_procrustes_phase_delta(
        &src_t.as_aligned_slice(),
        &tgt_t.as_aligned_slice(),
        &mut proc_out.as_aligned_slice_mut(),
    )
    .unwrap();
    assert_eq!(&proc_out.data[..8], &[32; 8]);

    // Verify exact binary dyadic reversible soft phase damping ([0.5, 2.0])
    let mut damp_buf = AlignedBuffer32::new([1.0_f32; 64]);
    eng.apply_soft_phase_damping(&mut damp_buf.as_aligned_slice_mut(), 0.5)
        .unwrap();
    eng.apply_soft_phase_damping(&mut damp_buf.as_aligned_slice_mut(), 2.0)
        .unwrap();
    assert_eq!(damp_buf.data[0].to_bits(), 1.0_f32.to_bits());

    let eng16 = DummyRopeEngine {
        head_dim: 16,
        num_heads: 4,
    };
    assert_eq!(eng16.resolve_autonomous_rot_dim(0x02, 0xFF, 0.1, 2.0), 2);
    assert_eq!(eng16.resolve_autonomous_rot_dim(0x00, 0x0F, 0.1, 1.0), 4);
    assert_eq!(eng16.resolve_autonomous_rot_dim(0x04, 0xFF, 0.2, 1.5), 6);
    assert_eq!(eng16.resolve_autonomous_rot_dim(0x08, 0xFF, 0.6, 1.0), 8);
}

#[test]
fn test_subhead_routing_and_visual_mse_bitmask() {
    let eng = DummyRopeEngine {
        head_dim: 12,
        num_heads: 8,
    };
    assert_eq!(eng.head_dim(), 12);
    let subhead_cfg = gcso_subhead_config_t {
        num_fact_heads: 2,
        num_logic_heads: 4,
        num_explore_heads: 2,
        fact_phase_gain: 1.0,
        logic_phase_gain: 1.0,
        explore_phase_gain: 1.2,
        ..Default::default()
    };
    let mut q_subhead = AlignedBuffer32::new([1.0_f32; 96]);
    let bank = AlignedBuffer64::<gcso_q7_t, 512>::splat(24);
    dispatch_phase_clamp_and_subhead_slot_ensemble_fused(
        &mut q_subhead.as_aligned_slice_mut(),
        &[16i8; 64],
        DEFAULT_PHASE_CLAMP_MAX_RAD,
        &subhead_cfg,
        &bank.relax_slice::<16>(),
        2,
        1,
        2,
        DEFAULT_PHASE_CLAMP_MAX_RAD,
        8,
        12,
        true,
    )
    .unwrap();
    assert_eq!(&q_subhead.data[0..10], &[1.0_f32; 10]);
    assert!((q_subhead.data[10] - 1.0).abs() > 1.0e-3);

    let mut frame_cur = AlignedBuffer32::new([1.0_f32; 48]);
    let frame_prv = AlignedBuffer32::new([1.0_f32; 48]);
    frame_cur.data[0..12].fill(2.0);
    let mut mask = gcso_paged_bitmask_t::default();
    let active_patches = detect_frame_mse_bitmask(
        &frame_cur.as_aligned_slice(),
        &frame_prv.as_aligned_slice(),
        4,
        12,
        0.1,
        &mut mask,
    )
    .unwrap();
    assert_eq!(active_patches, 1);
}
