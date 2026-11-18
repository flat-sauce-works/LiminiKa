// name: src/core/src/entropy_control/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
#[allow(unused_imports)]
use super::{closed_loop::*, controller::*, kernels::*};

#[test]
fn test_surrogate_and_moving_z_entropy() {
    let entropy = compute_sparse_surrogate_entropy(0.5, 0.5, DEFAULT_ENTROPY_EPS);
    assert!((entropy - LN_2).abs() < 1e-6);
    assert_eq!(
        compute_sparse_surrogate_entropy(1.0, 0.0, DEFAULT_ENTROPY_EPS),
        0.0
    );
    assert_eq!(
        compute_sparse_surrogate_entropy(-0.5, 0.5, DEFAULT_ENTROPY_EPS),
        0.0
    );

    let asym_h = compute_sparse_surrogate_entropy(0.8, 0.2, DEFAULT_ENTROPY_EPS);
    assert!(asym_h > 0.45 && asym_h < LN_2);

    let unnorm_h = compute_sparse_surrogate_entropy(1.6, 0.4, DEFAULT_ENTROPY_EPS);
    assert!((asym_h - unnorm_h).abs() < 1e-6);

    let tail_h = compute_sparse_surrogate_entropy(0.99, 0.01, DEFAULT_ENTROPY_EPS);
    assert!(tail_h > 0.03 && tail_h < 0.07);

    let h_tilde = compute_moving_z_score_entropy(0.5, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5);
    assert!(h_tilde > 0.5);
    assert_eq!(
        compute_moving_z_score_entropy(0.5, 0.1, 0.2, 0.1, 1e-6, 1.0, 0.5),
        0.5
    );
    assert_eq!(
        compute_moving_z_score_entropy(0.0, 0.8, 0.2, 0.1, 1e-6, 1.0, 0.5),
        0.0
    );
}

#[test]
fn test_entropy_controller_init_setters_and_reset() {
    let mut engine = EntropyController::default();
    assert_eq!(engine.state().bifurcation_threshold, 1.5);
    assert_eq!(engine.state().singularity_eps, DEFAULT_ENTROPY_EPS);
    assert_eq!(engine.state().sample_temperature, 1.0);
    assert_eq!(engine.hysteresis_band, DEFAULT_HYSTERESIS_BAND);
    assert_eq!(engine.tau_flux, DEFAULT_TAU_FLUX);
    assert!(engine.is_active());
    assert!(engine.is_normal_mode());
    assert_eq!(engine.as_bytes().len(), 128);

    engine.set_bifurcation_threshold(2.0);
    engine.set_bifurcation_threshold(-1.0);
    assert_eq!(engine.state().bifurcation_threshold, 2.0);

    engine.state.moving_z_entropy = 2.5;
    engine.state.active_branch_mode = EDBC_MODE_PHASE_CONJUGATE_REPULSION;
    engine.reset();
    assert_eq!(engine.moving_z_entropy(), 0.0);
    assert!(engine.is_normal_mode());

    let mut out_state = gcso_edbc_state_t::default();
    let h = EntropyClosedLoopCoordinator::eval_surrogate_and_step(
        &mut engine,
        0.5,
        0.5,
        0.8,
        0.2,
        0.1,
        1.0,
        0.5,
        &mut out_state,
    )
    .unwrap();
    assert!(h > 0.0 && out_state.moving_z_entropy == h);
}

#[test]
fn test_eval_stateful_modes_spike_ceiling_and_dwell_anti_chattering() {
    let mut engine = EntropyController::default().with_min_mode_dwell_steps(3);
    engine.set_bifurcation_threshold(1.0);
    engine.set_hysteresis_band(0.05);
    engine.set_tau_flux(0.5);
    let mut out_state = gcso_edbc_state_t::default();

    engine.state.moving_z_entropy = 2.0;
    engine.eval_stateful(10.0, &mut out_state).unwrap();
    assert_eq!(
        out_state.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert_eq!(engine.mode_dwell_steps, 1);

    let (emit_iframe, gop_len, can_skip, gate_scale) =
        engine.evaluate_adaptive_compute_and_gop_budget(6, 4, 32, 0.20);
    assert!(emit_iframe);
    assert_eq!(gop_len, 6);
    assert!(!can_skip);
    assert!((gate_scale - 1.0).abs() < 1e-5);

    engine.state.moving_z_entropy = 0.2;
    engine.eval_stateful(0.0, &mut out_state).unwrap();
    assert_eq!(
        out_state.active_branch_mode,
        EDBC_MODE_PHASE_CONJUGATE_REPULSION
    );
    assert_eq!(engine.mode_dwell_steps, 2);

    engine.eval_stateful(0.0, &mut out_state).unwrap();
    assert_eq!(out_state.active_branch_mode, EDBC_MODE_NORMAL_ENTROPY);

    let (emit_after, _gop_after, can_skip_after, _gate_after) =
        engine.evaluate_adaptive_compute_and_gop_budget(2, 4, 32, 0.25);
    assert!(!emit_after);
    assert!(can_skip_after);

    engine.reset();
    engine.eval_stateful(1.0e5, &mut out_state).unwrap();
    assert!(out_state.moving_z_entropy <= 1.0 + 1e-5);
    for _ in 0..15 {
        engine.eval_stateful(0.0, &mut out_state).unwrap();
    }
    assert!(engine.is_normal_mode());

    let (_flux, accel) = engine.eval_entropy_flux_acceleration();
    assert!(accel.is_finite());

    assert_eq!(
        engine.eval_stateful(f32::NAN, &mut out_state),
        Err(GCSO_ERROR_EDBC_SINGULARITY)
    );
}
