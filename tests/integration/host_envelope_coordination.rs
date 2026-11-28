// name: tests/integration/host_envelope_coordination.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-agnostic host-envelope coordination integration tests verifying Lv.0 `BoundarySignal`,
//! Lv.3 `EnvelopeState`, Lv.4 `RuntimeContext` 576 B snapshots, `EquilibriumState::join`
//! reconciliation, and `2 x 3` `AttributionTag` telemetry isolation without domain-specific terms.

use liminika_adapter::host::{EquilibriumConfig, EquilibriumState};
use liminika_core::drivers::{
    ENVELOPE_BREACH_NONE, ENVELOPE_BREACH_PRESSURE, ENVELOPE_BREACH_UNDEFINED_FAULT,
    ENVELOPE_FLAG_EMA_INITIALIZED, ENVELOPE_LOCK_EMA_PAUSED, ENVELOPE_LOCK_FAULT_ISOLATED,
    EnvelopeBounds, EnvelopeState,
};
use liminika_core::foundation::Pod;
use liminika_core::liminal::{
    AttributionTag, BOUNDARY_SIGNAL_FLAG_FAULT, BOUNDARY_SIGNAL_FLAG_OBSERVED, BoundarySignal,
    HOST_IMMUTABLE_PINNED_COMPUTE_MASK, Lacuna, evaluate_lacuna_partition_branchless,
};
use liminika_core::runtime::{
    Config, CoreTelemetryBank, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES, RuntimeContext,
};

fn build_nominal_signal(pressure_q15: u16, chatter_q7: u8, epoch_id: u32) -> BoundarySignal {
    let part =
        evaluate_lacuna_partition_branchless(0b0000_0011, &[64, 60, 0, 0, 0, 0, 0, 0], &[8; 8], 32);
    BoundarySignal::from_partition(part, pressure_q15, 0, chatter_q7, epoch_id)
}

#[test]
fn scenario1_nominal_signal_roundtrip_and_branchless_lacuna_derivation() {
    let sig = build_nominal_signal(8_192, 6, 1);
    assert_eq!(core::mem::size_of::<BoundarySignal>(), 32);
    assert_eq!(core::mem::align_of::<BoundarySignal>(), 32);
    assert_eq!(sig.lacuna_code(), Lacuna::SelfRef.code());
    assert!(sig.is_observed());
    assert!(!sig.is_fault());
    assert_eq!(sig.reserved, [0; 2]);

    // Lossless 32-byte Pod byte roundtrip across sensor/actuator channels.
    let bytes = sig.as_bytes();
    assert_eq!(bytes.len(), 32);
    let restored = BoundarySignal::read_unaligned_from(bytes, 0).expect("pod read");
    assert_eq!(restored, sig);
    assert_eq!(restored.lacuna_code(), Lacuna::SelfRef.code());

    let bounds = EnvelopeBounds::new();
    let mut env = EnvelopeState::new();
    let fb = env.step(&bounds, &restored);
    assert_eq!(fb.contraction_q7, 128);
    assert_eq!(fb.breach_mask, ENVELOPE_BREACH_NONE);
    assert_eq!(env.violation_epochs, 0);
}

#[test]
fn scenario2_pressure_and_undefined_breaches_contract_and_protect_pin_floor() {
    let mut bounds = EnvelopeBounds::new();
    bounds.immutable_compute_pin_mask = HOST_IMMUTABLE_PINNED_COMPUTE_MASK | (1_u64 << 3);
    bounds.pressure_ceiling_q15 = 24_000;
    bounds.contraction_gain_q7 = 32;
    bounds.min_aperture_q7 = 24;

    let mut env = EnvelopeState::new();
    env.apply_host_pin_floor(1_u64 << 5);
    let initial_floor = env.baseline_pinned_compute_mask;
    assert_ne!(initial_floor & HOST_IMMUTABLE_PINNED_COMPUTE_MASK, 0);
    assert_ne!(initial_floor & (1_u64 << 5), 0);

    // Pressure exceedance contracts aperture immediately and increments violation_epochs.
    let high_pressure = build_nominal_signal(30_000, 8, 2);
    let fb_press = env.step(&bounds, &high_pressure);
    assert_eq!(fb_press.contraction_q7, 96);
    assert_ne!(fb_press.breach_mask & ENVELOPE_BREACH_PRESSURE, 0);
    assert_eq!(env.violation_epochs, 1);
    assert_eq!(
        env.baseline_pinned_compute_mask & initial_floor,
        initial_floor
    );
    assert_ne!(env.baseline_pinned_compute_mask & (1_u64 << 3), 0);

    // Terminal Undefined fault drives contraction to 0 (fail-closed) without weakening pin floor.
    let mut fault_sig = build_nominal_signal(10_000, 4, 3);
    fault_sig.signal_flags |= BOUNDARY_SIGNAL_FLAG_FAULT | BOUNDARY_SIGNAL_FLAG_OBSERVED;
    assert_eq!(fault_sig.lacuna_code(), Lacuna::Undefined.code());
    let fb_fault = env.step(&bounds, &fault_sig);
    assert_eq!(fb_fault.contraction_q7, 0);
    assert_ne!(fb_fault.breach_mask & ENVELOPE_BREACH_UNDEFINED_FAULT, 0);
    assert_ne!(fb_fault.lock_flags & ENVELOPE_LOCK_FAULT_ISOLATED, 0);
    assert_eq!(env.violation_epochs, 2);
    assert_eq!(
        env.baseline_pinned_compute_mask & initial_floor,
        initial_floor
    );

    // Even a zeroed bounds request cannot strip the latched permanent pin floor.
    let relaxed_bounds = EnvelopeBounds {
        immutable_compute_pin_mask: 0,
        ..bounds
    };
    let _ = env.step(&relaxed_bounds, &build_nominal_signal(8_000, 2, 4));
    assert_eq!(
        env.baseline_pinned_compute_mask & initial_floor,
        initial_floor
    );
    assert_ne!(
        env.baseline_pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
        0
    );
}

#[test]
fn scenario3_calm_recovery_respects_min_aperture_floor_and_recovers_monotonically() {
    let mut bounds = EnvelopeBounds::new();
    bounds.pressure_ceiling_q15 = 20_000;
    bounds.contraction_gain_q7 = 40;
    bounds.recovery_gain_q7 = 16;
    bounds.min_aperture_q7 = 32;

    let mut env = EnvelopeState::new();
    let stress = build_nominal_signal(28_000, 4, 1);
    // Repeated non-fault violations clamp strictly at `min_aperture_q7`.
    for ep in 1..=6_u32 {
        let mut s = stress;
        s.epoch_id = ep;
        let fb = env.step(&bounds, &s);
        assert!(fb.contraction_q7 >= bounds.min_aperture_q7);
    }
    assert_eq!(env.contraction_q7, bounds.min_aperture_q7);
    assert_eq!(env.calm_epochs, 0);

    // Consecutive calm epochs monotonically recover aperture back to unit scale (128).
    let mut prev_c = env.contraction_q7;
    for ep in 7..=14_u32 {
        let calm = build_nominal_signal(8_000, 2, ep);
        let fb = env.step(&bounds, &calm);
        assert_eq!(fb.breach_mask, ENVELOPE_BREACH_NONE);
        assert!(fb.contraction_q7 >= prev_c);
        prev_c = fb.contraction_q7;
    }
    assert_eq!(env.contraction_q7, 128);
    assert_eq!(env.calm_epochs, 8);
}

#[test]
fn scenario4_two_node_576b_snapshots_join_monotonic_pins_and_winner_temp_locks() {
    // Node A (earlier generation): sets compute bit 2, aperture bit 4, slot bit 5, action bit 3,
    // and engages temporary locks (`unobserved_locked = 1`, `irreversible_step_lock = 1`).
    let mut node_a = RuntimeContext::new(Config::new()).expect("node a");
    let mut bounds_a = EnvelopeBounds::new();
    bounds_a.immutable_compute_pin_mask = HOST_IMMUTABLE_PINNED_COMPUTE_MASK | (1_u64 << 2);
    bounds_a.immutable_aperture_pin_mask |= 1_u8 << 4;
    bounds_a.immutable_slot_pin_mask |= 1_u8 << 5;
    bounds_a.immutable_action_pin_mask |= 1_u8 << 3;
    let _ = node_a.step_envelope(&bounds_a, &build_nominal_signal(9_000, 4, 1));
    node_a.set_unobserved_locked(true);
    node_a.set_irreversible_step_lock(true);
    node_a.substrate_mut().step_counter = 10;

    let mut snap_a = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req_a = 0_usize;
    node_a
        .serialize_to_bytes(&mut snap_a, &mut req_a)
        .expect("serialize a");
    assert_eq!(req_a, 576);

    // Node B (incoming winner): sets compute bit 6, aperture bit 6, slot bit 7, action bit 4,
    // and keeps temporary locks cleared (`unobserved_locked = 0`, `irreversible_step_lock = 0`).
    let mut node_b = RuntimeContext::new(Config::new()).expect("node b");
    let mut bounds_b = EnvelopeBounds::new();
    bounds_b.immutable_compute_pin_mask = HOST_IMMUTABLE_PINNED_COMPUTE_MASK | (1_u64 << 6);
    bounds_b.immutable_aperture_pin_mask |= 1_u8 << 6;
    bounds_b.immutable_slot_pin_mask |= 1_u8 << 7;
    bounds_b.immutable_action_pin_mask |= 1_u8 << 4;
    let _ = node_b.step_envelope(&bounds_b, &build_nominal_signal(7_500, 3, 2));
    node_b.set_unobserved_locked(false);
    node_b.set_irreversible_step_lock(false);
    node_b.substrate_mut().step_counter = 20;

    let mut snap_b = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req_b = 0_usize;
    node_b
        .serialize_to_bytes(&mut snap_b, &mut req_b)
        .expect("serialize b");
    assert_eq!(req_b, 576);

    let cfg = EquilibriumConfig::default();
    let mut eq_a = EquilibriumState::new(cfg);
    eq_a.update_runtime_snapshot_from_container(&snap_a)
        .expect("install snap a");
    let mut eq_b = EquilibriumState::new(cfg);
    eq_b.update_runtime_snapshot_from_container(&snap_b)
        .expect("install snap b");

    let join_rep = eq_a.join(&eq_b).expect("join snapshots");
    assert!(join_rep.snapshot_updated);

    let merged = eq_a.snapshot_envelope_state().expect("merged envelope");
    // All 5 pin-floor masks are monotonically OR-merged across both nodes.
    let expected_compute = HOST_IMMUTABLE_PINNED_COMPUTE_MASK | (1_u64 << 2) | (1_u64 << 6);
    assert_eq!(
        merged.baseline_pinned_compute_mask & expected_compute,
        expected_compute
    );
    assert_eq!(
        merged.pinned_compute_mask & expected_compute,
        expected_compute
    );
    assert_eq!(
        merged.pinned_aperture_mask & ((1_u8 << 4) | (1_u8 << 6)),
        (1_u8 << 4) | (1_u8 << 6)
    );
    assert_eq!(
        merged.pinned_slot_mask & ((1_u8 << 5) | (1_u8 << 7)),
        (1_u8 << 5) | (1_u8 << 7)
    );
    assert_eq!(
        merged.pinned_action_mode_mask & ((1_u8 << 3) | (1_u8 << 4)),
        (1_u8 << 3) | (1_u8 << 4)
    );
    // Temporary locks follow the winner (node B = 0) instead of being OR-merged from node A.
    assert_eq!(merged.unobserved_locked, 0);
    assert_eq!(merged.irreversible_step_lock, 0);
}

#[test]
fn scenario5_envelope_attribution_tags_count_cleanly_without_driver_pollution() {
    let lim_env = AttributionTag::LIMINAL_ENVELOPE;
    let stig_env = AttributionTag::STIGMERGY_ENVELOPE;
    assert_eq!(lim_env.raw(), 4);
    assert_eq!(stig_env.raw(), 5);
    assert!(lim_env.is_envelope() && !lim_env.is_system() && !lim_env.is_ego());
    assert!(stig_env.is_envelope() && !stig_env.is_system() && !stig_env.is_ego());

    let mut bank = CoreTelemetryBank::default();
    let part = evaluate_lacuna_partition_branchless(0x03, &[64, 60, 0, 0, 0, 0, 0, 0], &[8; 8], 32);
    let tension = liminika_core::drivers::DistortionTension::default();
    bank.record_two_by_two_step(lim_env.raw(), false, &part, &tension, 0);
    bank.record_two_by_two_step(lim_env.raw(), false, &part, &tension, 0);
    bank.record_two_by_two_step(stig_env.raw(), false, &part, &tension, 0);

    let stack = liminika_core::runtime::CoreStateStack::new();
    let horizon = liminika_core::runtime::EpistemicHorizonReading::default();
    let snap = bank.snapshot(&stack, &horizon, 0, 0, 0);
    assert_eq!(snap.attribution_counts[0], 0);
    assert_eq!(snap.attribution_counts[1], 0);
    assert_eq!(snap.attribution_counts[2], 0);
    assert_eq!(snap.attribution_counts[3], 0);
    assert_eq!(snap.attribution_counts[4], 2);
    assert_eq!(snap.attribution_counts[5], 1);
}

#[test]
fn scenario6_two_stream_online_ema_convergence_pause_resume_and_bounds_join() {
    // Stream A: slower shift (ema_shift = 4) and tighter ceiling; experiences a breach then dwells.
    let mut ctx_a = RuntimeContext::new(Config::new()).expect("ctx_a");
    let bounds_a = EnvelopeBounds {
        pressure_ceiling_q15: 25_000,
        pressure_floor_q15: 4_000,
        min_baseline_pressure_q15: 4_000,
        max_baseline_pressure_q15: 22_000,
        ema_resume_dwell_target: 3,
        ema_shift: 4,
        ..EnvelopeBounds::new()
    };
    ctx_a.configure_envelope_bounds(bounds_a).expect("bounds_a");
    let _ = ctx_a.step_envelope(&bounds_a, &build_nominal_signal(10_000, 4, 1));
    assert_eq!(ctx_a.envelope().baseline_pressure_q15, 10_000);
    assert_ne!(
        ctx_a.envelope().envelope_flags & ENVELOPE_FLAG_EMA_INITIALIZED,
        0
    );
    let fb_breach = ctx_a.step_envelope(&bounds_a, &build_nominal_signal(29_000, 4, 2));
    assert_ne!(fb_breach.breach_mask & ENVELOPE_BREACH_PRESSURE, 0);
    assert_ne!(fb_breach.lock_flags & ENVELOPE_LOCK_EMA_PAUSED, 0);
    assert_eq!(ctx_a.envelope().ema_resume_dwell, 0);
    assert_eq!(ctx_a.envelope().baseline_pressure_q15, 10_000);
    for ep in 3..=4_u32 {
        let fb = ctx_a.step_envelope(&bounds_a, &build_nominal_signal(14_000, 4, ep));
        assert_ne!(fb.lock_flags & ENVELOPE_LOCK_EMA_PAUSED, 0);
        assert_eq!(ctx_a.envelope().baseline_pressure_q15, 10_000);
    }
    let fb_resume = ctx_a.step_envelope(&bounds_a, &build_nominal_signal(14_000, 4, 5));
    assert_eq!(fb_resume.lock_flags & ENVELOPE_LOCK_EMA_PAUSED, 0);
    assert!(ctx_a.envelope().baseline_pressure_q15 > 10_000);
    ctx_a.substrate_mut().step_counter = 12;

    // Stream B (winner): faster shift (ema_shift = 2) converging upward on a steady stream.
    let mut ctx_b = RuntimeContext::new(Config::new()).expect("ctx_b");
    let bounds_b = EnvelopeBounds {
        pressure_ceiling_q15: 26_500,
        pressure_floor_q15: 5_000,
        min_baseline_pressure_q15: 5_000,
        max_baseline_pressure_q15: 24_000,
        ema_resume_dwell_target: 2,
        ema_shift: 2,
        ..EnvelopeBounds::new()
    };
    ctx_b.configure_envelope_bounds(bounds_b).expect("bounds_b");
    let _ = ctx_b.step_envelope(&bounds_b, &build_nominal_signal(8_000, 2, 1));
    for ep in 2..=8_u32 {
        let fb = ctx_b.step_envelope(&bounds_b, &build_nominal_signal(16_000, 2, ep));
        assert_eq!(fb.lock_flags & ENVELOPE_LOCK_EMA_PAUSED, 0);
    }
    assert!(ctx_b.envelope().baseline_pressure_q15 > 14_000);
    ctx_b.substrate_mut().step_counter = 24;

    let mut snap_a = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut snap_b = [0_u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req = 0_usize;
    ctx_a.serialize_to_bytes(&mut snap_a, &mut req).expect("a");
    ctx_b.serialize_to_bytes(&mut snap_b, &mut req).expect("b");

    let cfg = EquilibriumConfig::default();
    let mut eq_a = EquilibriumState::new(cfg);
    eq_a.update_runtime_snapshot_from_container(&snap_a)
        .expect("install a");
    let mut eq_b = EquilibriumState::new(cfg);
    eq_b.update_runtime_snapshot_from_container(&snap_b)
        .expect("install b");
    assert_eq!(eq_a.snapshot_envelope_bounds(), Some(bounds_a));
    assert_eq!(eq_b.snapshot_envelope_bounds(), Some(bounds_b));

    let rep = eq_a.join(&eq_b).expect("join streams");
    assert!(rep.snapshot_updated);
    // Winner (Stream B) EnvelopeBounds @ 416..448 is inherited with valid invariants.
    let joined_bounds = eq_a.snapshot_envelope_bounds().expect("joined bounds");
    assert_eq!(joined_bounds, bounds_b);
    assert!(joined_bounds.validate_invariants());

    // Roundtrip back into a fresh RuntimeContext restores both EnvelopeState and EnvelopeBounds.
    let container_576 = eq_a
        .runtime_snapshot_container_bytes()
        .expect("576B container");
    let mut restored = RuntimeContext::new(Config::new()).expect("restored");
    restored
        .deserialize_from_bytes(&container_576)
        .expect("deserialize");
    assert_eq!(*restored.envelope_bounds(), bounds_b);
    assert_eq!(
        restored.envelope().baseline_pressure_q15,
        ctx_b.envelope().baseline_pressure_q15
    );
}
