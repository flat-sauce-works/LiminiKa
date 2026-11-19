// name: src/core/src/foundation/rope/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::foundation::aligned::{AlignedBuffer128, AlignedBuffer16, AlignedBuffer32};
use crate::foundation::pod::Pod;
use crate::foundation::scalar_math::{branchless_sincos_q7, branchless_sqrt_f32};
use core::f32::consts::PI;

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
fn test_phase_steering_engine_creation_aligned_dispatch_and_pod_layout() {
    assert!(PhaseSteeringEngine::new(128, 32).is_ok());
    assert!(PhaseSteeringEngine::new(127, 32).is_err());
    assert!(PhaseSteeringEngine::new(0, 32).is_err());
    assert!(PhaseSteeringEngine::new(128, 0).is_err());

    let def = PhaseSteeringEngine::default()
        .with_phase_clamp_max_rad(0.18)
        .with_phase_step_min_rad(0.01);
    assert_eq!(def.head_dim(), 128);
    assert_eq!(def.num_heads(), 32);
    assert!((def.phase_clamp_max_rad() - 0.18).abs() < 1e-6);
    assert!((def.phase_step_min_rad() - 0.01).abs() < 1e-6);
    assert_eq!(def.as_bytes().len(), 16);

    let eng_small = PhaseSteeringEngine::new(8, 2).unwrap();
    let mut q128 = AlignedBuffer128::new([
        1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
    ]);
    let p16 = AlignedBuffer16::new([64i8; 16]);
    eng_small
        .apply_phase_steering_aligned::<128>(
            &mut q128.as_aligned_slice_mut(),
            &p16.as_aligned_slice(),
        )
        .unwrap();
    assert!((q128.data[0] - 0.0).abs() < 1e-4);
    assert!((q128.data[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_qdps_filter_and_swar_scaling() {
    let engine = PhaseSteeringEngine::new(128, 16).unwrap();
    let mut raw_buf =
        AlignedBuffer16::new([1i8, 5, -2, 10, -128, 4, -1, 8, 6, -6, 12, -12, 0, 2, -3, 16]);
    let mut aligned = raw_buf.as_aligned_slice_mut();

    assert!(engine.phase_step_filter(&mut aligned, 0.1).is_ok());
    let res = aligned.as_slice();
    assert_eq!(res[0], 0);
    assert_eq!(res[1], 5);
    assert_eq!(res[2], 0);
    assert_eq!(res[3], 10);
    assert_eq!(res[4], -128);
    assert_eq!(res[5], 0);
    assert_eq!(res[6], 0);
    assert_eq!(res[7], 8);
    assert_eq!(res[15], 16);
}

#[test]
fn test_lazy_unwrap_override() {
    let engine = PhaseSteeringEngine::new(4, 1).unwrap();
    let mut q_buf = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
    let c_buf = AlignedBuffer32::new([PI * 0.5]);

    engine
        .lazy_unwrap_override(&mut q_buf.as_aligned_slice_mut(), &c_buf.as_aligned_slice())
        .unwrap();
    assert!((q_buf.data[0] - 0.0).abs() < 1e-4);
    assert!((q_buf.data[1] - 1.0).abs() < 1e-4);
    assert!((q_buf.data[2] - (-1.0)).abs() < 1e-4);
    assert!((q_buf.data[3] - 0.0).abs() < 1e-4);
}

#[test]
fn test_slerp_norm_guard_logit_shift_and_procrustes() {
    let engine = PhaseSteeringEngine::new(4, 2).unwrap();
    let mut t_buf = AlignedBuffer32::new([2.0f32; 8]);
    engine
        .slerp_norm_guard_stable(&mut t_buf.as_aligned_slice_mut(), 1.0, 1.0)
        .unwrap();
    let norm_sq: f32 = t_buf.data.iter().map(|v| v * v).sum();
    assert!((branchless_sqrt_f32(norm_sq) - 1.0).abs() < 1e-5);

    let mut logits = AlignedBuffer32::new([1.0f32; 8]);
    let mut phases_arr = [0i8; 16];
    phases_arr[0] = 32;
    phases_arr[1] = 64;
    let phases = AlignedBuffer16::new(phases_arr);
    engine
        .fused_logit_shift(
            &mut logits.as_aligned_slice_mut(),
            &phases.as_aligned_slice(),
        )
        .unwrap();
    let expected_shift = (48.0 * PI) / 128.0;
    for &val in &logits.data {
        assert!((val - (1.0 + expected_shift)).abs() < 1e-4);
    }

    let engine_8h = PhaseSteeringEngine::new(16, 8).unwrap();
    let mut src_8h = AlignedBuffer32::new([0.0f32; 128]);
    for pair in src_8h.data.chunks_exact_mut(2) {
        pair[0] = 1.0;
        pair[1] = 0.0;
    }
    let mut tgt_8h = src_8h;
    let p_64 = AlignedBuffer16::new([64i8; 16]);
    engine_8h
        .apply_phase_steering(&mut tgt_8h.as_aligned_slice_mut(), &p_64.as_aligned_slice())
        .unwrap();
    let mut out_8h = AlignedBuffer16::new([0i8; 16]);
    engine_8h
        .compute_procrustes_phase_delta(
            &src_8h.as_aligned_slice(),
            &tgt_8h.as_aligned_slice(),
            &mut out_8h.as_aligned_slice_mut(),
        )
        .unwrap();
    for h in 0..8 {
        assert_eq!(out_8h.data[h], 64);
    }

    let (s_pi, c_pi) = branchless_sincos_q7(-128);
    assert!(s_pi.abs() < 1e-6);
    assert!((c_pi - (-1.0)).abs() < 1e-6);
}
