// name: src/core/src/phase_steering/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
#[allow(unused_imports)]
use super::{closed_loop::*, engine::*};
use crate::traits::aligned::{AlignedBuffer128, AlignedBuffer16, AlignedBuffer32};
use crate::traits::gateway::{ObjectiveFact, SubjectivePersona, SystemConstraint};
use crate::traits::primitives::{branchless_sincos_q7, ATTRACTOR_SLOT_KNOWLEDGE};
use core::f32::consts::PI;

#[test]
fn test_phase_steering_engine_creation_aligned_dispatch_and_pod_layout() {
    assert!(PhaseSteeringEngine::new(128, 32).is_ok());
    assert!(PhaseSteeringEngine::new(127, 32).is_err());
    assert!(PhaseSteeringEngine::new(0, 32).is_err());
    assert!(PhaseSteeringEngine::new(128, 0).is_err());

    let def = PhaseSteeringEngine::default()
        .with_ripa_clamp_max_rad(0.18)
        .with_qdps_min_step_rad(0.01);
    assert_eq!(def.head_dim(), 128);
    assert_eq!(def.num_heads(), 32);
    assert!((def.ripa_clamp_max_rad() - 0.18).abs() < 1e-6);
    assert!((def.qdps_min_step_rad() - 0.01).abs() < 1e-6);
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

    assert!(engine.qdps_filter_step(&mut aligned, 0.1).is_ok());
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
fn test_slerp_norm_guard_logit_shift_procrustes_and_gateway_signal() {
    let engine = PhaseSteeringEngine::new(4, 2).unwrap();
    let mut t_buf = AlignedBuffer32::new([2.0f32; 8]);
    engine
        .slerp_norm_guard_stable(&mut t_buf.as_aligned_slice_mut(), 1.0, 1.0)
        .unwrap();
    let norm_sq: f32 = t_buf.data.iter().map(|v| v * v).sum();
    assert!((crate::traits::primitives::branchless_sqrt_f32(norm_sq) - 1.0).abs() < 1e-5);

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

    let engine_gw = PhaseSteeringEngine::new(4, 16).unwrap();
    let sys = SystemConstraint::new((), 0.1, 0.0, 0xFF);
    let fact = ObjectiveFact::new([40i8; 64], 2.5, 0x0010_0000, 42);
    let persona = SubjectivePersona::new([0i8; 64], 0.0, 0, 0);
    let mut q_data = [0.0f32; 64];
    for pair in q_data.chunks_exact_mut(2) {
        pair[0] = 1.0;
    }
    let mut q_gw = AlignedBuffer32::new(q_data);
    let mut trail_gw = gcso_pointer_trail_t::default();
    let (frame, report) = engine_gw
        .steer_gateway_signal_safe(
            &(sys, fact, persona),
            &mut trail_gw,
            64,
            &mut q_gw.as_aligned_slice_mut(),
            None,
            None,
        )
        .unwrap();
    assert_eq!(frame.target_anchor_id, 42);
    assert_eq!(report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
    assert_eq!(trail_gw.target_anchor_id, 42);
}
