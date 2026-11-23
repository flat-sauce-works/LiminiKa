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
    assert!(ctx
        .serialize_to_bytes(&mut buffer, &mut required_size)
        .is_ok());

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
        ctx.system_spine().pinned_compute_mask & 0b0101_0000,
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
        apply_latent_horizon_slots, CoreStateStack, GatewayFrame, HorizonDwellState,
        HorizonTriggerInput,
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

    // Step 1: Destabilized (step 1 of dwell, high repulsion) -> only FastSlot slides in!
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

    // Step 3: Quiescent (step 1 of dwell) -> FastSlot slides out with cleanup, HysteresisSlot holds!
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
