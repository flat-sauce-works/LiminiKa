// name: src/core/src/runtime/context/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::liminal::LMNK_ERR_CONTAINER_CORRUPTED;
use crate::*;

#[test]
fn test_config_validation_and_builder() {
    let config = Config::new()
        .with_head_dim(128)
        .with_num_heads(32)
        .with_paged_block_size(32)
        .with_hysteresis_band(0.2)
        .with_bifurcation_threshold(1.4)
        .with_repulsion_threshold(2);

    assert!(config.validate().is_ok());

    let invalid_config = config.with_head_dim(127);
    assert_eq!(invalid_config.validate(), Err(LMNK_ERR_INVALID_ARGUMENT));
    assert_eq!(
        Config::from_raw(RuntimeConfig {
            rotary_dim: 256,
            ..config.raw
        })
        .validate(),
        Err(LMNK_ERR_INVALID_ARGUMENT)
    );
    assert_eq!(
        Config::from_raw(RuntimeConfig {
            rope_layout: 2,
            ..config.raw
        })
        .validate(),
        Err(LMNK_ERR_INVALID_ARGUMENT)
    );
}

#[test]
fn test_runtime_context_lifecycle_and_serialization() {
    let config = Config::new().with_repulsion_threshold(3);
    let mut ctx = RuntimeContext::new(config).unwrap();
    assert_eq!(ctx.slot_enabled_mask(), 0x00);
    assert_eq!(ctx.slot_pinned_mask(), 0x00);
    assert_eq!(ctx.slot_repulsion_mask(), 0x00);
    assert_eq!(ctx.slot_sandbox_mask(), 0x00);

    let pinned_lane = ctx
        .substrate_mut()
        .allocate_lane(&WorkbenchSlotSpec::explicit(
            usize::from(ATTRACTOR_SLOT_PERSONA) as u8,
            true,
            0,
            2.5,
            [12; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(pinned_lane, ATTRACTOR_SLOT_PERSONA);
    let rep_lane = ctx
        .substrate_mut()
        .allocate_lane(&WorkbenchSlotSpec::auto_repulsion(
            1.0,
            [-8; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    let sb_lane = ctx
        .substrate_mut()
        .allocate_lane(&WorkbenchSlotSpec::auto_sandbox(
            0.75,
            [6; MAX_ATTENTION_HEADS],
        ))
        .unwrap();
    assert_eq!(
        ctx.find_dominant_domain().0,
        u32::from(ATTRACTOR_SLOT_PERSONA)
    );
    let expected_enabled = (1u8 << pinned_lane) | (1u8 << rep_lane) | (1u8 << sb_lane);
    let expected_pinned = 1u8 << pinned_lane;
    let expected_rep = 1u8 << rep_lane;
    let expected_sb = 1u8 << sb_lane;

    let mut trail = PointerTrail::default();
    assert!(ctx.step_token(42, None, None, Some(&mut trail)).is_ok());
    assert_eq!(ctx.step_counter(), 1);
    assert_eq!(trail.current_ptr, 42);

    let mut required_size = 0usize;
    assert_eq!(ctx.serialize_to_bytes(&mut [], &mut required_size), Ok(()));
    assert_eq!(required_size, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);

    let mut buffer = alloc::vec![0u8; required_size];
    assert!(
        ctx.serialize_to_bytes(&mut buffer, &mut required_size)
            .is_ok()
    );

    // Host-pinned Spine bits must survive untrusted snapshot deserialization intact.
    ctx.apply_host_spine_pin_floor(0b0101_0000);
    assert!(ctx.reset().is_ok());
    assert_eq!(ctx.step_counter(), 0);
    assert_eq!(ctx.slot_enabled_mask(), 0x00);
    ctx.apply_host_spine_pin_floor(0b0101_0000);
    assert!(ctx.deserialize_from_bytes(&buffer).is_ok());
    assert_eq!(ctx.step_counter(), 1);
    assert_eq!(ctx.substrate().repulsion_threshold, 3);
    assert_eq!(ctx.slot_enabled_mask(), expected_enabled);
    assert_eq!(ctx.slot_pinned_mask(), expected_pinned);
    assert_eq!(ctx.slot_repulsion_mask(), expected_rep);
    assert_eq!(ctx.slot_sandbox_mask(), expected_sb);
    assert!((ctx.slot_weights()[usize::from(ATTRACTOR_SLOT_PERSONA)] - 2.5).abs() < 1e-5);
    assert_eq!(
        ctx.envelope().pinned_compute_mask & 0b0101_0000,
        0b0101_0000
    );

    // Verify IEEE 802.3 CRC32 payload tamper detection
    buffer[200] ^= 0xFF;
    assert_eq!(
        ctx.deserialize_from_bytes(&buffer),
        Err(LMNK_ERR_CONTAINER_CORRUPTED)
    );
}

#[test]
fn test_runtime_context_capabilities_and_mapped_storage() {
    let config = Config::new();
    let ctx = RuntimeContext::new(config).unwrap();

    assert_eq!(
        ctx.abi_version(),
        (
            LMNK_ABI_VERSION_MAJOR,
            LMNK_ABI_VERSION_MINOR,
            LMNK_ABI_VERSION_PATCH,
        )
    );
    assert_eq!(ctx.abi_version_string(), env!("CARGO_PKG_VERSION"));
    assert_ne!(ctx.query_capability() & LMNK_CAP_SWAR_BASELINE, 0);
    assert_ne!(ctx.query_capability() & LMNK_CAP_INPLACE_CONTEXT_INIT, 0);

    assert_eq!(
        ctx.get_track_offset(0).unwrap(),
        core::mem::size_of::<LmnkContainerHeader>() as u64
    );
}

#[test]
fn test_runtime_context_vertical_piercing_and_dynamic_slide_in_out() {
    use crate::drivers::EGO_TRIGGER_BOUNDARY_CHATTER;
    use crate::foundation::rope::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
    use crate::prelude::{AlignedBuffer32, ObjectiveFact, SubjectivePersona, SystemConstraint};

    struct MockGeom;
    impl RopeSteeringTarget for MockGeom {
        fn rope_geometry(&self) -> RopeGeometryDescriptor {
            RopeGeometryDescriptor {
                head_dim: 128,
                num_heads: 32,
                rotary_dim: 32,
                layout: RopePairLayout::Interleaved,
            }
        }
    }

    struct MockSlideInSlot {
        invoked_count: u32,
    }
    impl LatentHorizonSlot for MockSlideInSlot {
        fn horizon_slot_mask(&self) -> u32 {
            1 << 3
        }

        fn on_slide_in_modulate(
            &mut self,
            frame: &mut GatewayFrame,
            stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            self.invoked_count += 1;
            stack.liminal.active_aperture_mask |= 1 << 7;
            frame.phase_deltas.fill(24);
        }
    }

    struct MockEscalator {
        calls: u32,
    }
    impl HorizonEscalationProvider for MockEscalator {
        fn try_escalate_horizon(
            &mut self,
            _req: &HorizonEscalationRequest,
            slot_phase_out: &mut [PhaseQ7; MAX_ATTENTION_HEADS],
        ) -> Result<Option<f32>> {
            self.calls += 1;
            slot_phase_out.fill(18);
            Ok(Some(1.25))
        }
    }

    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let mut q_buf = AlignedBuffer32::new([1.0_f32; 128 * 32]);
    let mut trail = PointerTrail::default();
    let signal = (
        SystemConstraint::new((), 0.1, 0.0, 0xFF),
        ObjectiveFact::new([12i8; 64], 1.5, 0x1000, 7),
        SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
    );
    let mut slot = MockSlideInSlot { invoked_count: 0 };
    let mut escalator = MockEscalator { calls: 0 };

    // 1. Nominal coherent state -> no slide-in, no escalation.
    let (_f1, _r1, in1, out1, esc1) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &MockGeom,
            &mut slot,
            &mut escalator,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(in1, 0);
    assert_eq!(out1, 0);
    assert!(!esc1);
    assert_eq!(slot.invoked_count, 0);
    assert_eq!(ctx.active_horizon_slots_mask(), 0);
    assert_eq!(ctx.escalated_workbench_lane(), None);
    assert_eq!(ctx.slot_enabled_mask(), 0);

    // 2. Destabilize via Core Stack Slot 2 Ego trigger -> slot slides in & external escalator leases an anonymous lane.
    ctx.core_layers_mut().ego.last_intervened_mask = EGO_TRIGGER_BOUNDARY_CHATTER;
    let (f2, _r2, in2, out2, esc2) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &MockGeom,
            &mut slot,
            &mut escalator,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(in2, 1 << 3);
    assert_eq!(out2, 0);
    assert!(esc2);
    assert_eq!(slot.invoked_count, 1);
    assert_eq!(escalator.calls, 1);
    let leased_lane = ctx.escalated_workbench_lane().expect("leased lane");
    assert_ne!(ctx.slot_enabled_mask() & (1 << leased_lane), 0);
    assert_eq!(f2.phase_deltas[0], 24);
    assert_ne!(
        ctx.orchestrator_status().liminal_threshold_mask & (1 << 7),
        0
    );
    assert_eq!(ctx.active_horizon_slots_mask(), 1 << 3);

    // 3. Stabilize -> slot automatically retracts (slides out) and leased lane returns to free pool.
    ctx.core_layers_mut().ego.last_intervened_mask = 0;
    ctx.last_boundary_delta_mut().stationary_shortcut_ready = 1;
    let (_f3, _r3, in3, out3, esc3) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &MockGeom,
            &mut slot,
            &mut escalator,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(in3, 0);
    assert_eq!(out3, 1 << 3);
    assert!(!esc3);
    assert_eq!(ctx.active_horizon_slots_mask(), 0);
    assert_eq!(ctx.escalated_workbench_lane(), None);
    assert_eq!(ctx.slot_enabled_mask() & (1 << leased_lane), 0);
    assert!(ctx.horizon_dwell().quiescent_dwell_steps >= 1);
}

#[test]
fn test_latent_horizon_slot_tuple_isolation_dwell_and_slide_out_cleanup() {
    use {
        CoreStateStack, GatewayFrame, HorizonDwellState, HorizonTriggerInput,
        apply_latent_horizon_slots,
    };

    struct FastSlot {
        slid_out_calls: u32,
    }
    impl LatentHorizonSlot for FastSlot {
        fn horizon_slot_mask(&self) -> u32 {
            1 << 1
        }

        fn should_slide_in(&self, trigger: &HorizonTriggerInput) -> bool {
            trigger.repulsion_severity > 0.4
        }

        fn on_slide_in_modulate(
            &mut self,
            frame: &mut GatewayFrame,
            _stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            frame.phase_deltas[1] = 32;
        }

        fn on_slide_out(
            &mut self,
            frame: &mut GatewayFrame,
            _stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            self.slid_out_calls += 1;
            frame.phase_deltas[1] = 0;
        }
    }

    struct HysteresisSlot {
        modulate_calls: u32,
    }
    impl LatentHorizonSlot for HysteresisSlot {
        fn horizon_slot_mask(&self) -> u32 {
            1 << 4
        }

        fn min_slide_in_dwell(&self) -> u16 {
            2
        }

        fn min_slide_out_dwell(&self) -> u16 {
            2
        }

        fn on_slide_in_modulate(
            &mut self,
            frame: &mut GatewayFrame,
            stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            self.modulate_calls += 1;
            stack.liminal.active_aperture_mask |= 1 << 4;
            frame.phase_deltas[4] = 48;
        }
    }

    let mut pair = (
        FastSlot { slid_out_calls: 0 },
        HysteresisSlot { modulate_calls: 0 },
    );
    let mut active_mask = 0u32;
    let mut dwell = HorizonDwellState::default();
    let mut frame = GatewayFrame::default();
    let mut stack = CoreStateStack::new();

    // Step 1: Destabilized (step 1 of dwell, high repulsion) -> only FastSlot slides in.
    let mut t1 = HorizonTriggerInput {
        repulsion_severity: 0.6,
        ..HorizonTriggerInput::default()
    };
    let (u1, q1) = dwell.step_branchless(t1.is_destabilized(), t1.stationary_shortcut_ready);
    t1.unstable_dwell_steps = u1;
    t1.quiescent_dwell_steps = q1;
    let (in1, out1) =
        apply_latent_horizon_slots(&mut pair, &mut active_mask, &t1, &mut frame, &mut stack);
    assert_eq!(in1, 1 << 1);
    assert_eq!(out1, 0);
    assert_eq!(active_mask, 1 << 1);
    assert_eq!(pair.1.modulate_calls, 0);

    // Step 2: Destabilized persists (step 2 of dwell) -> HysteresisSlot now slides in independently.
    let mut t2 = HorizonTriggerInput {
        repulsion_severity: 0.6,
        ..HorizonTriggerInput::default()
    };
    let (u2, q2) = dwell.step_branchless(t2.is_destabilized(), t2.stationary_shortcut_ready);
    t2.unstable_dwell_steps = u2;
    t2.quiescent_dwell_steps = q2;
    let (in2, out2) =
        apply_latent_horizon_slots(&mut pair, &mut active_mask, &t2, &mut frame, &mut stack);
    assert_eq!(in2, 1 << 4);
    assert_eq!(out2, 0);
    assert_eq!(active_mask, (1 << 1) | (1 << 4));
    assert_eq!(pair.1.modulate_calls, 1);

    // Step 3: Quiescent (step 1 of dwell) -> FastSlot slides out with cleanup, HysteresisSlot holds.
    let mut t3 = HorizonTriggerInput {
        stationary_shortcut_ready: true,
        ..HorizonTriggerInput::default()
    };
    let (u3, q3) = dwell.step_branchless(t3.is_destabilized(), t3.stationary_shortcut_ready);
    t3.unstable_dwell_steps = u3;
    t3.quiescent_dwell_steps = q3;
    let (in3, out3) =
        apply_latent_horizon_slots(&mut pair, &mut active_mask, &t3, &mut frame, &mut stack);
    assert_eq!(in3, 0);
    assert_eq!(out3, 1 << 1);
    assert_eq!(active_mask, 1 << 4);
    assert_eq!(pair.0.slid_out_calls, 1);
    assert_eq!(frame.phase_deltas[1], 0);

    // Step 4: Quiescent persists (step 2 of dwell) -> HysteresisSlot retracts.
    let mut t4 = HorizonTriggerInput {
        stationary_shortcut_ready: true,
        ..HorizonTriggerInput::default()
    };
    let (u4, q4) = dwell.step_branchless(t4.is_destabilized(), t4.stationary_shortcut_ready);
    t4.unstable_dwell_steps = u4;
    t4.quiescent_dwell_steps = q4;
    let (in4, out4) =
        apply_latent_horizon_slots(&mut pair, &mut active_mask, &t4, &mut frame, &mut stack);
    assert_eq!(in4, 0);
    assert_eq!(out4, 1 << 4);
    assert_eq!(active_mask, 0);
}

#[test]
fn test_residual_spillover_slot_cache_promotion_eviction_and_escalation() {
    use crate::drivers::EGO_TRIGGER_INTERNAL_CONFLICT;
    use crate::foundation::rope::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
    use crate::prelude::{
        AlignedBuffer32, DefaultResidualSpilloverSlot, ObjectiveFact, SubjectivePersona,
        SystemConstraint,
    };
    use crate::stigmergy::DissipativeOverflowReport;

    struct Geom;
    impl RopeSteeringTarget for Geom {
        fn rope_geometry(&self) -> RopeGeometryDescriptor {
            RopeGeometryDescriptor {
                head_dim: 128,
                num_heads: 32,
                rotary_dim: 32,
                layout: RopePairLayout::Interleaved,
            }
        }
    }

    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let mut q_buf = AlignedBuffer32::new([1.0_f32; 128 * 32]);
    let mut trail = PointerTrail::default();
    let signal = (
        SystemConstraint::new((), 0.2, 0.0, 0xFF),
        ObjectiveFact::new([20i8; 64], 1.5, 0x3000, 17),
        SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
    );

    let mut spillover = DefaultResidualSpilloverSlot::default();
    let mut fallback_esc = ();
    spillover.stage_overflow_report(&DissipativeOverflowReport {
        overflowed: true,
        residual_words: [0x1212_1212_1212_1212; 8],
        ..DissipativeOverflowReport::default()
    });

    // 1. Step 1 destabilized -> held back by slide_in_dwell (2).
    ctx.core_layers_mut().ego.last_intervened_mask = EGO_TRIGGER_INTERNAL_CONFLICT;
    let (_f1, _, in1, _, _) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &Geom,
            &mut spillover,
            &mut fallback_esc,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(in1, 0);
    assert_eq!(spillover.spillover_modulations, 0);

    // 2. Step 2 sustained destabilization -> spillover slides in (cache miss -> orthogonal elevation).
    let (_f2, _, in2, _, _) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &Geom,
            &mut spillover,
            &mut fallback_esc,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert_eq!(in2, spillover.slot_bit);
    assert_eq!(spillover.spillover_modulations, 1);

    // 3. Pollution guard: failed resolution stores nothing in experience cache.
    let verified_phases = [0x12i8; 64];
    let (stored_bad, _) = spillover.commit_verified_experience(0, &verified_phases, 0.9, false);
    assert!(!stored_bad);
    assert_eq!(spillover.cache_stores, 0);

    // 4. Verified resolution promotes solution into Tier-1 RAM cache.
    let (stored_ok, _) = spillover.commit_verified_experience(0, &verified_phases, 0.9, true);
    assert!(stored_ok);
    assert_eq!(spillover.cache_stores, 1);

    // 5. Subsequent destabilization hits Tier-1 cache AND feeds HorizonEscalationProvider lane 7.
    let mut escalator = spillover.clone();
    let (_f3, _, _, _, esc3) = ctx
        .pierce_and_orchestrate_step(
            &signal,
            &Geom,
            &mut spillover,
            &mut escalator,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        )
        .unwrap();
    assert!(esc3);
    assert_eq!(spillover.cache_hits, 1);

    // 6. Contradiction penalty halves pheromone and evicts below floor on repeated penalty.
    spillover.evict_pheromone_floor = 0.8;
    let (_, evicted) = spillover.commit_verified_experience(0, &verified_phases, 0.0, false);
    assert_eq!(evicted, 1);
    assert!(spillover.cache_evictions >= 1);
}

#[test]
fn test_spine_policy_and_decode_control_four_lacunae_and_guard_relaxation() {
    struct CustomCalmSignal {
        step_idx: u16,
        pressure: SystemPressureReading,
        greedy: bool,
    }

    impl SpineDecodeSignal for CustomCalmSignal {
        fn pressure_reading(&self) -> SystemPressureReading {
            self.pressure
        }
        fn exit_entropy_lacuna(&self) -> Lacuna {
            Lacuna::SelfRef
        }
        fn observed_exit_entropy_q15(&self) -> u16 {
            13_107 // ~0.40 (< 0.55 reward split & < 0.72 guard relax bound)
        }
        fn boundary_sensitivity_q7(&self) -> u8 {
            SPINE_SENTINEL_AUTO_U8
        }
        fn decode_step_index(&self) -> u16 {
            self.step_idx
        }
        fn repulsion_active(&self) -> bool {
            false
        }
        fn greedy_sampling(&self) -> bool {
            self.greedy
        }
    }

    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let active = 0x00FF_u64;
    let host_base = 0x0081_u64;
    let rev = BitCut64::split(active, host_base).outside;
    ctx.compute_pool_mut().reanchor_baseline_pressure(8_000);
    ctx.compute_pool_mut()
        .set_slot_masks(active, host_base, 0, rev);
    ctx.set_baseline_pin_mask(host_base);
    ctx.system_spine_mut().set_max_observe_stride(4);
    ctx.system_spine_mut().guard_hold_tokens = 2;

    // 1. Guarded turn arms active_mask transiently without modifying baseline_pinned_compute_mask.
    ctx.begin_turn_spine_policy(true, false);
    assert!(ctx.system_spine().turn_guard_active());
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, host_base);
    assert_eq!(ctx.envelope().pinned_compute_mask, active);
    assert_eq!(ctx.system_spine().last_executed_mask, active);

    let calm_pressure = SystemPressureReading {
        frame_time_pressure_q15: 8_000,
        bandwidth_pressure_q15: 8_000,
        l3_load_pressure_q15: 4_000,
        irreversible_step: 0,
        _pad: 0,
    };

    // Step 0 & Step 1: hold window (2) keeps concrete execution; step 0 requires ObserveFullLogits.
    // Step 2: Stage 3 sees concrete_run_steps == 2 & calm_concrete_streak == 2, relaxes guard pin
    // back to host_base, enters whole-token bypass, and schedules SkipToDeviceArgmax for step 3.
    let plan0 = ctx.step_decode_control(
        &CustomCalmSignal {
            step_idx: 0,
            pressure: calm_pressure,
            greedy: true,
        },
        false,
    );
    assert_eq!(
        plan0.next_observe_mode_enum(),
        SpineLogitsObserveMode::ObserveFullLogits
    );
    assert!(!plan0.is_whole_token_bypass());
    assert_eq!(plan0.event_flags & SPINE_STEP_EVENT_BOUNDARY_CHURN, 0);

    let plan1 = ctx.step_decode_control(
        &CustomCalmSignal {
            step_idx: 1,
            pressure: calm_pressure,
            greedy: true,
        },
        false,
    );
    assert!(!plan1.is_whole_token_bypass());
    assert_eq!(
        plan1.event_flags & SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED,
        0
    );

    let plan2 = ctx.step_decode_control(
        &CustomCalmSignal {
            step_idx: 2,
            pressure: calm_pressure,
            greedy: true,
        },
        false,
    );
    assert_ne!(
        plan2.event_flags & SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED,
        0
    );
    assert!(ctx.system_spine().is_transient_guard_relaxed());
    assert!(!ctx.system_spine().turn_guard_active());
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, host_base);
    assert_eq!(ctx.envelope().pinned_compute_mask, host_base);
    assert!(plan2.is_whole_token_bypass());
    assert_eq!(plan2.dispatch.execute_mask, active);
    assert_eq!(
        plan2.next_observe_mode_enum(),
        SpineLogitsObserveMode::SkipToDeviceArgmax
    );
    assert!(plan2.effective_observe_stride >= 2);

    // 2. Step 3 receives Lacuna::Unobserved -> skips D2H entropy update without dirtying entropy.
    let saved_ent_q15 = ctx.system_spine().last_observed_entropy_q15;
    let plan3 = ctx.step_decode_control(
        &SpineDecodeInput::unobserved(calm_pressure, 3, 8, false, true),
        false,
    );
    assert_ne!(plan3.event_flags & SPINE_STEP_EVENT_BOUNDARY_CHURN, 0);
    assert_ne!(
        plan3.event_flags & SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED,
        0
    );
    assert_eq!(ctx.system_spine().last_observed_entropy_q15, saved_ent_q15);
    assert!(plan3.is_whole_token_bypass());

    // 3. Exterior surge triggers instability recovery & re-escalates relaxed guard without polluting base.
    let plan4 = ctx.step_decode_control(
        &SpineDecodeInput::from_observed_entropy_f32(calm_pressure, 4, 0.92, 96, false, true),
        false,
    );
    assert_ne!(plan4.event_flags & SPINE_STEP_EVENT_INSTABILITY_RECOVERY, 0);
    assert_ne!(plan4.event_flags & SPINE_STEP_EVENT_GUARD_REESCALATED, 0);
    assert!(!plan4.is_whole_token_bypass());
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, host_base);
    assert_eq!(ctx.envelope().pinned_compute_mask, active);

    // 4. Undefined (NaN) rejects reward commit and never overwrites last_observed_entropy_q15.
    let ent_before_nan = ctx.system_spine().last_observed_entropy_q15;
    let ph_before_nan = ctx.compute_pool().slot_pheromone_q7[2];
    let nan_input =
        SpineDecodeInput::from_observed_entropy_f32(calm_pressure, 5, f32::NAN, 8, false, true);
    assert_eq!(nan_input.exit_entropy_lacuna(), Lacuna::Undefined);
    let plan5 = ctx.step_decode_control(&nan_input, false);
    assert_ne!(plan5.event_flags & SPINE_STEP_EVENT_INSTABILITY_RECOVERY, 0);
    assert_ne!(plan5.event_flags & SPINE_STEP_EVENT_BOUNDARY_CHURN, 0);
    assert_eq!(ctx.system_spine().last_observed_entropy_q15, ent_before_nan);
    assert_eq!(ctx.compute_pool().slot_pheromone_q7[2], ph_before_nan);

    // 5. Telemetry snapshot reflects counters unconditionally under Standard verbosity.
    let snap = ctx.telemetry_snapshot();
    assert_eq!(snap.concrete_token_count, 4);
    assert_eq!(snap.abstract_token_count, 2);
    assert_eq!(snap.boundary_churn_count, 2);
    assert_eq!(snap.logits_skip_count, 1);
    assert_eq!(snap.guard_relaxation_count, 1);
    assert_eq!(snap.instability_recovery_count, 2);

    // 6. Reward uses completed-step `last_executed_mask` rather than unconditionally full active_mask.
    ctx.begin_turn_spine_policy(false, false);
    assert_eq!(ctx.envelope().pinned_compute_mask, host_base);
    ctx.system_spine_mut().last_executed_mask = 0x0001_u64;
    let slot0_before = ctx.compute_pool().slot_pheromone_q7[0];
    let slot2_before = ctx.compute_pool().slot_pheromone_q7[2];
    let _ = ctx.step_decode_control(
        &SpineDecodeInput::from_observed_entropy_f32(calm_pressure, 0, 0.70, 8, false, true),
        false,
    );
    assert!(ctx.compute_pool().slot_pheromone_q7[0] >= slot0_before);
    assert!(ctx.compute_pool().slot_pheromone_q7[2] < slot2_before);
}

#[test]
fn test_config_spine_policy_propagation_and_configure_spine() {
    let cfg = Config::new()
        .with_guard_hold_tokens(5)
        .with_max_observe_stride(6);
    assert_eq!(cfg.raw.guard_hold_tokens, 5);
    assert_eq!(cfg.raw.max_observe_stride, 6);

    // 1. RuntimeContext::new + reset preserve configured spine policy and host pin floor.
    let mut ctx = RuntimeContext::new(cfg).unwrap();
    assert_eq!(ctx.system_spine().guard_hold_tokens, 5);
    assert_eq!(ctx.system_spine().max_observe_stride, 6);
    ctx.configure_spine(0x0081, 3, 0, false);
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, 0x0081);
    assert_eq!(ctx.envelope().pinned_compute_mask, 0x0081);
    assert_eq!(ctx.system_spine().guard_hold_tokens, 3);
    assert_eq!(ctx.system_spine().max_observe_stride, 1);
    assert!(!ctx.system_spine().is_pre_bypass_enabled());
    // Passing 0 for baseline_pinned_compute_mask retains existing host floor monotonically.
    ctx.configure_spine(0, 7, 4, true);
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, 0x0081);
    assert_eq!(ctx.system_spine().guard_hold_tokens, 7);
    assert_eq!(ctx.system_spine().max_observe_stride, 4);
    assert!(ctx.system_spine().is_pre_bypass_enabled());

    ctx.reset().unwrap();
    assert_eq!(ctx.envelope().baseline_pinned_compute_mask, 0x0081);
    assert_eq!(ctx.system_spine().guard_hold_tokens, 7);
    assert_eq!(ctx.system_spine().max_observe_stride, 4);

    // 2. RuntimeContext::write_in_place honors zero-init `max_observe_stride = 0` -> `1` and `guard_hold_tokens`.
    let mut raw_zero_stride = cfg.raw;
    raw_zero_stride.max_observe_stride = 0;
    let mut uninit = core::mem::MaybeUninit::<RuntimeContext>::uninit();
    // SAFETY: `uninit` is 64-byte aligned, writable, and non-aliased.
    unsafe {
        RuntimeContext::write_in_place(uninit.as_mut_ptr(), Config::from_raw(raw_zero_stride))
            .unwrap();
        let inplace = uninit.assume_init_ref();
        assert_eq!(inplace.system_spine().guard_hold_tokens, 5);
        assert_eq!(inplace.system_spine().max_observe_stride, 1);
        assert_eq!(*inplace.envelope_bounds(), EnvelopeBounds::new());
    }
}

#[test]
fn test_decode_control_envelope_ema_unobserved_stride_and_snapshot_bounds_roundtrip() {
    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let custom_bounds = EnvelopeBounds {
        pressure_ceiling_q15: 24_000,
        pressure_floor_q15: 4_000,
        min_baseline_pressure_q15: 4_000,
        max_baseline_pressure_q15: 22_000,
        ema_resume_dwell_target: 3,
        ema_shift: 2,
        ..EnvelopeBounds::new()
    };
    ctx.configure_envelope_bounds(custom_bounds)
        .expect("valid bounds");
    assert_eq!(*ctx.envelope_bounds(), custom_bounds);

    // Invalid bounds rejected fail-closed without mutating active bounds.
    let mut bad_bounds = custom_bounds;
    bad_bounds.ema_shift = 8;
    assert_eq!(
        ctx.configure_envelope_bounds(bad_bounds),
        Err(LMNK_ERR_INVALID_ARGUMENT)
    );
    assert_eq!(*ctx.envelope_bounds(), custom_bounds);

    // Three-baseline separation: ComputePool Gauge-Zero baseline stays untouched by Envelope EMA.
    ctx.configure_compute_pool(0x00FF, 0, 0x00FE, 9_500);
    let pool_base_before = ctx.compute_pool().baseline_pressure_q15;
    assert_eq!(pool_base_before, 9_500);

    let pressure = SystemPressureReading::default();
    ctx.begin_turn_spine_policy(false, false);

    // 1. Observed calm steps seed and converge EnvelopeState::baseline_pressure_q15.
    let _ = ctx.step_decode_control(
        &SpineDecodeInput::from_observed_entropy_f32(pressure, 0, 0.30, 8, false, true),
        false,
    );
    let seeded_base = ctx.envelope().baseline_pressure_q15;
    assert!((9_800..=9_900).contains(&seeded_base));
    assert_ne!(
        ctx.envelope().envelope_flags & ENVELOPE_FLAG_EMA_INITIALIZED,
        0
    );
    assert_eq!(ctx.envelope().lock_flags() & ENVELOPE_LOCK_EMA_PAUSED, 0);

    for step in 1..=6u16 {
        let _ = ctx.step_decode_control(
            &SpineDecodeInput::from_observed_entropy_f32(pressure, step, 0.50, 8, false, true),
            false,
        );
    }
    let tracked_base = ctx.envelope().baseline_pressure_q15;
    assert!(tracked_base > seeded_base);
    assert_eq!(ctx.compute_pool().baseline_pressure_q15, pool_base_before);

    // 2. Unobserved stride never triggers false breach or pulls baseline toward zero.
    let _ = ctx.step_decode_control(
        &SpineDecodeInput::unobserved(pressure, 7, 8, false, true),
        false,
    );
    assert_eq!(ctx.envelope().breach_mask, ENVELOPE_BREACH_NONE);
    assert_eq!(ctx.envelope().baseline_pressure_q15, tracked_base);
    assert_eq!(ctx.envelope().ema_resume_dwell, 0);
    assert_ne!(ctx.envelope().lock_flags() & ENVELOPE_LOCK_EMA_PAUSED, 0);

    // 3. High-entropy breach above custom ceiling (24_000 / 32_768 ~ 0.732) triggers recovery,
    //    pauses EMA, and resets dwell; 3 consecutive calm steps resume EMA tracking.
    let breach_plan = ctx.step_decode_control(
        &SpineDecodeInput::from_observed_entropy_f32(pressure, 8, 0.76, 8, false, true),
        false,
    );
    assert_ne!(
        breach_plan.event_flags & SPINE_STEP_EVENT_INSTABILITY_RECOVERY,
        0
    );
    assert_ne!(ctx.envelope().breach_mask & ENVELOPE_BREACH_PRESSURE, 0);
    assert_eq!(ctx.envelope().baseline_pressure_q15, tracked_base);
    assert_eq!(ctx.envelope().ema_resume_dwell, 0);
    assert_ne!(ctx.envelope().lock_flags() & ENVELOPE_LOCK_EMA_PAUSED, 0);

    for step in 9..=10u16 {
        let _ = ctx.step_decode_control(
            &SpineDecodeInput::from_observed_entropy_f32(pressure, step, 0.25, 8, false, true),
            false,
        );
        assert_eq!(ctx.envelope().baseline_pressure_q15, tracked_base);
        assert_ne!(ctx.envelope().lock_flags() & ENVELOPE_LOCK_EMA_PAUSED, 0);
    }
    let _ = ctx.step_decode_control(
        &SpineDecodeInput::from_observed_entropy_f32(pressure, 11, 0.25, 8, false, true),
        false,
    );
    assert_eq!(ctx.envelope().ema_resume_dwell, 3);
    assert_eq!(ctx.envelope().lock_flags() & ENVELOPE_LOCK_EMA_PAUSED, 0);
    assert!(ctx.envelope().baseline_pressure_q15 < tracked_base);

    // 4. RuntimeSnapshot roundtrip preserves both EnvelopeState and custom EnvelopeBounds.
    let mut snap = [0u8; LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES];
    let mut req = 0usize;
    ctx.serialize_to_bytes(&mut snap, &mut req).unwrap();
    assert_eq!(req, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);

    let mut restored = RuntimeContext::new(Config::new()).unwrap();
    restored.deserialize_from_bytes(&snap).unwrap();
    assert_eq!(*restored.envelope_bounds(), custom_bounds);
    assert_eq!(
        restored.envelope().baseline_pressure_q15,
        ctx.envelope().baseline_pressure_q15
    );
    assert_eq!(
        restored.envelope().ema_resume_dwell,
        ctx.envelope().ema_resume_dwell
    );
    assert_eq!(
        restored.envelope().envelope_flags,
        ctx.envelope().envelope_flags
    );
}
