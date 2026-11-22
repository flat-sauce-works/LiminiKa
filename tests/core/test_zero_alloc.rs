// name: tests/core/test_zero_alloc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test enforcing zero dynamic heap allocation on the LiminiKa Hot Path.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use liminika_core::abi::{
    lmnk_config_init_default, lmnk_config_t, lmnk_context_align, lmnk_context_create,
    lmnk_context_deinit_in_place, lmnk_context_destroy, lmnk_context_init_in_place,
    lmnk_context_reset, lmnk_context_size, lmnk_context_step_token,
    lmnk_phase_steering_apply_fused_safe, lmnk_pointer_trail_t,
    lmnk_substrate_resolve_trail_in_swarm, LmnkContextHandle, LMNK_OK,
};
use liminika_core::foundation::{
    AlignedBuffer16, AlignedBuffer32, DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD,
};
use liminika_core::stigmergy::StigmergicReconstitutionReport;

struct ZeroAllocGuardAllocator;

thread_local! {
    static ENFORCE_ZERO_ALLOC: Cell<bool> = const { Cell::new(false) };
}

// SAFETY: Delegates directly to `std::alloc::System` while enforcing zero-allocation during Hot Path windows on the active test thread.
unsafe impl GlobalAlloc for ZeroAllocGuardAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ENFORCE_ZERO_ALLOC.with(Cell::get) {
            ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
            panic!(
                "Zero-Alloc Violation! Dynamic allocation detected in Hot-Path: {:?}",
                layout
            );
        }
        // SAFETY: Forwarded directly to the system allocator with caller-provided `layout`.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` and `layout` originate from `System.alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static A: ZeroAllocGuardAllocator = ZeroAllocGuardAllocator;

#[test]
fn test_context_step_token_zero_allocation_guarantee() {
    // 1. Setup Context and aligned buffers in Cold Path (allocations allowed here).
    let mut config = lmnk_config_t::default();
    // SAFETY: `config` is a valid mutable stack reference.
    let status = unsafe { lmnk_config_init_default(&mut config) };
    assert_eq!(status, LMNK_OK);

    let mut ctx_handle: LmnkContextHandle = std::ptr::null_mut();
    // SAFETY: `config` is initialized and `ctx_handle` is a valid out-pointer.
    let status = unsafe { lmnk_context_create(&config, &mut ctx_handle) };
    assert_eq!(status, LMNK_OK);
    assert!(!ctx_handle.is_null());

    let mut query_buf = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let mut key_buf = AlignedBuffer32::<f32, 128>::new([0.25; 128]);
    let mut trail_out = lmnk_pointer_trail_t::default();

    const HEAD_DIM: usize = 128;
    const NUM_HEADS: usize = 16;
    let mut steer_q =
        AlignedBuffer32::<f32, { HEAD_DIM * NUM_HEADS }>::new([1.0; HEAD_DIM * NUM_HEADS]);
    let mut steer_k =
        AlignedBuffer32::<f32, { HEAD_DIM * NUM_HEADS }>::new([0.5; HEAD_DIM * NUM_HEADS]);
    let mut phases = [0i8; 64];
    for (i, slot) in phases.iter_mut().enumerate() {
        *slot = ((i as i8).wrapping_mul(7)).wrapping_sub(32);
    }
    let phase_buf = AlignedBuffer16::<i8, 64>::new(phases);
    let mut swarm_report = StigmergicReconstitutionReport::default();

    // 2. Enable Strict Zero-Allocation Guard for Hot Path.
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));

    // 3. Execute Hot Path step loop (verified zero-allocation).
    for token_id in 0..100 {
        // SAFETY: `ctx_handle` is valid, `query_buf`/`key_buf` are disjoint 32-byte-aligned
        // 128-element buffers, and `trail_out` is 64-byte aligned.
        let step_status = unsafe {
            lmnk_context_step_token(
                ctx_handle,
                token_id,
                query_buf.as_mut_slice().as_mut_ptr(),
                key_buf.as_mut_slice().as_mut_ptr(),
                &mut trail_out,
            )
        };
        assert_eq!(step_status, LMNK_OK);

        // SAFETY: `steer_q` and `steer_k` are disjoint 32B-aligned tensors, `phase_buf` is 16B-aligned.
        let steer_status = unsafe {
            lmnk_phase_steering_apply_fused_safe(
                steer_q.as_mut_slice().as_mut_ptr(),
                steer_k.as_mut_slice().as_mut_ptr(),
                phase_buf.as_slice().as_ptr(),
                HEAD_DIM,
                NUM_HEADS,
                DEFAULT_PHASE_CLAMP_MAX_RAD,
            )
        };
        assert_eq!(steer_status, LMNK_OK);

        // SAFETY: `trail_out`, `phase_buf`, and `swarm_report` are valid, aligned, and disjoint.
        let swarm_status = unsafe {
            lmnk_substrate_resolve_trail_in_swarm(
                &mut trail_out,
                u64::from(token_id) + 1,
                phase_buf.as_slice().as_ptr(),
                0xFF,
                64,
                ((DEFAULT_PHASE_STEP_MIN_RAD / core::f32::consts::PI) * 128.0) as u8,
                16,
                4,
                0.5,
                0.2,
                &mut swarm_report,
            )
        };
        assert_eq!(swarm_status, LMNK_OK);
    }

    // 4. Disable Guard before Cold Path cleanup.
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
    // SAFETY: `ctx_handle` was created by `lmnk_context_create` and is destroyed once.
    let destroy_status = unsafe { lmnk_context_destroy(ctx_handle) };
    assert_eq!(destroy_status, LMNK_OK);
}

#[test]
fn test_zero_alloc_in_place_context_lifecycle() {
    #[repr(C, align(64))]
    struct AlignedContextBacking([u8; 8192]);

    let mut config = lmnk_config_t::default();
    let mut backing = AlignedContextBacking([0u8; 8192]);
    let mut q_buf = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let mut k_buf = AlignedBuffer32::<f32, 128>::new([0.25; 128]);
    let mut trail_out = lmnk_pointer_trail_t::default();

    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));

    // SAFETY: Stack backing satisfies `lmnk_context_size()` and 64-byte alignment without heap ops.
    unsafe {
        assert_eq!(lmnk_config_init_default(&mut config), LMNK_OK);
        let req_size = lmnk_context_size();
        let req_align = lmnk_context_align();
        assert_eq!(req_align, 64);
        assert!(req_size > 0 && req_size <= backing.0.len());
        assert_eq!(
            lmnk_context_init_in_place(&config, backing.0.as_mut_ptr(), req_size),
            LMNK_OK
        );
        let ctx_handle = backing
            .0
            .as_mut_ptr()
            .cast::<liminika_core::abi::LmnkContextOpaque>();
        assert_eq!(
            lmnk_context_step_token(
                ctx_handle,
                77,
                q_buf.as_mut_slice().as_mut_ptr(),
                k_buf.as_mut_slice().as_mut_ptr(),
                &mut trail_out,
            ),
            LMNK_OK
        );
        assert_eq!(trail_out.step_count, 1);
        assert_eq!(lmnk_context_reset(ctx_handle), LMNK_OK);
        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);
        assert!(backing.0[..req_size].iter().all(|&b| b == 0));
        assert_eq!(
            lmnk_context_deinit_in_place(ctx_handle),
            liminika_core::abi::LMNK_ERR_INVALID_STATE
        );

        let mut topo = liminika_core::abi::lmnk_topology_matrix_t::default();
        assert_eq!(
            liminika_core::abi::lmnk_topology_get_matrix(&mut topo),
            LMNK_OK
        );
        assert!(topo.verify_crc32());

        let anchor = liminika_core::PhaseAnchorRecord::new_empty(
            7,
            liminika_core::stigmergy::ATTRACTOR_SLOT_PERSONA,
            0x1000,
        );
        #[repr(C, align(64))]
        struct AlignedContainerBacking([u8; 512]);
        let mut container_buf = AlignedContainerBacking([0u8; 512]);
        let written = liminika_core::LmnkContainerView::write_standard_sections_to_bytes(
            liminika_core::LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            &[anchor],
            &[],
            None,
            None,
            &[],
            1_774_900_000,
            &mut container_buf.0,
        )
        .expect("container write succeeds");
        let parsed = liminika_core::LmnkContainerView::parse(&container_buf.0[..written])
            .expect("zero-alloc parse succeeds");
        assert_eq!(parsed.phase_anchor_count(), 1);
    }

    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
}

#[test]
fn test_runtime_context_pierce_and_slide_in_zero_alloc() {
    use liminika_core::drivers::EGO_TRIGGER_BOUNDARY_CHATTER;
    use liminika_core::foundation::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
    use liminika_core::runtime::{
        CoreStateStack, GatewayFrame, HorizonTriggerInput, LatentHorizonSlot, ObjectiveFact,
        SubjectivePersona, SystemConstraint,
    };
    use liminika_core::{Config, RuntimeContext};

    struct TestGeom;
    impl RopeSteeringTarget for TestGeom {
        fn rope_geometry(&self) -> RopeGeometryDescriptor {
            RopeGeometryDescriptor {
                head_dim: 128,
                num_heads: 32,
                rotary_dim: 32,
                layout: RopePairLayout::Interleaved,
            }
        }
    }

    struct ZeroAllocSlideSlot;
    impl LatentHorizonSlot for ZeroAllocSlideSlot {
        fn horizon_slot_mask(&self) -> u32 {
            1 << 2
        }

        fn on_slide_in_modulate(
            &mut self,
            frame: &mut GatewayFrame,
            stack: &mut CoreStateStack,
            _trigger: &HorizonTriggerInput,
        ) {
            stack.liminal.active_aperture_mask |= 1 << 5;
            frame.phase_deltas[0] = 16;
        }
    }

    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let mut q_buf = AlignedBuffer32::<f32, { 128 * 32 }>::new([1.0_f32; 128 * 32]);
    let mut trail = lmnk_pointer_trail_t::default();
    let signal = (
        SystemConstraint::new((), 0.2, 0.0, 0xFF),
        ObjectiveFact::new([10i8; 64], 1.2, 0x2000, 11),
        SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
    );
    let mut slot = ZeroAllocSlideSlot;
    let mut no_esc = ();

    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));
    for step in 0..32 {
        ctx.core_layers_mut().ego.last_intervened_mask = if (step & 1) == 0 {
            EGO_TRIGGER_BOUNDARY_CHATTER
        } else {
            0
        };
        ctx.last_boundary_delta_mut().stationary_shortcut_ready = u8::from((step & 1) != 0);
        let res = ctx.pierce_and_orchestrate_step(
            &signal,
            &TestGeom,
            &mut slot,
            &mut no_esc,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        );
        assert!(res.is_ok());
    }
    let snap = ctx.telemetry_snapshot();
    assert_eq!(snap.spine.total_steps, 32);

    // Verify ResidualSpilloverHorizonSlot<4> + Tier-1 cache stays strictly zero-alloc.
    let mut spillover = liminika_core::ResidualSpilloverHorizonSlot::<4>::new();
    let mut no_esc2 = ();
    for _ in 0..8 {
        ctx.core_layers_mut().ego.last_intervened_mask = EGO_TRIGGER_BOUNDARY_CHATTER;
        let res = ctx.pierce_and_orchestrate_step(
            &signal,
            &TestGeom,
            &mut spillover,
            &mut no_esc2,
            &mut trail,
            Some(&mut q_buf.as_aligned_slice_mut()),
        );
        assert!(res.is_ok());
    }
    let _ = spillover.commit_verified_experience(0, &[12i8; 64], 0.9, true);
    let _ = ctx.pierce_and_orchestrate_step(
        &signal,
        &TestGeom,
        &mut spillover,
        &mut no_esc2,
        &mut trail,
        Some(&mut q_buf.as_aligned_slice_mut()),
    );
    assert!(spillover.cache_hits >= 1);

    // Verify Free-by-Default 8-lane pool allocator + escalation remain strictly zero-alloc.
    let mut wb_substrate = liminika_core::StigmergicSubstrate::new();
    for i in 0..8u8 {
        let lane = wb_substrate
            .allocate_lane(&liminika_core::WorkbenchSlotSpec::auto_free(
                0.2 + f32::from(i) * 0.1,
                [i as i8 + 1; 64],
            ))
            .unwrap();
        assert_eq!(lane, i);
    }
    wb_substrate.pin_lane(0, true).unwrap();
    let ev_lane = wb_substrate
        .allocate_lane(&liminika_core::WorkbenchSlotSpec::auto_sandbox(
            0.9, [16; 64],
        ))
        .unwrap();
    assert_eq!(ev_lane, 1);
    assert!(wb_substrate.settle_sandbox_lane(ev_lane).unwrap());
    let _ = wb_substrate.decay_and_reclaim_lanes(0.5, 0.25);
    let _ = wb_substrate.release_unpinned_lanes();
    let mut esc_lane = None;
    let esc_req = liminika_core::runtime::HorizonEscalationRequest {
        trigger: liminika_core::runtime::HorizonTriggerInput {
            repulsion_severity: 0.8,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut mock_esc = liminika_core::DefaultResidualSpilloverSlot::default();
    let _ = wb_substrate.apply_horizon_escalation(&esc_req, &mut esc_lane, &mut mock_esc);

    // Verify label-free AnonymousComputePool dispatch, whole-token bypass, Venn partition,
    // SystemSpine isolation, DistortionDelta P-Frame chain, and DistortionTension remain zero-alloc.
    ctx.system_spine_mut().pin_compute_slots(0b0001);
    ctx.compute_pool_mut()
        .set_slot_masks(0b1111, 0b0001, 0, 0b1110);
    let mut dispatch = ctx.step_compute_dispatch(liminika_core::drivers::SystemPressureReading {
        frame_time_pressure_q15: 20000,
        bandwidth_pressure_q15: 10000,
        l3_load_pressure_q15: 9000,
        irreversible_step: 0,
        _pad: 0,
    });
    ctx.commit_compute_exit_reward(dispatch.execute_mask, 96);
    let _ = ctx.compute_pool_mut().evaluate_whole_token_steering_bypass(
        liminika_core::drivers::TokenBypassObservation {
            irreversible_step: 0,
            liminal_stationary: 1,
            boundary_sensitivity_q7: 16,
            repulsion_active: 0,
            exit_entropy_rising: 0,
            ego_active_boundary: 0,
            min_calm_dwell: 2,
            _pad: 0,
        },
        &mut dispatch,
    );
    let part = liminika_core::evaluate_lacuna_partition_branchless(
        0b0000_0011,
        &[80, 72, 0, 0, 96, 88, 0, 0],
        &[4; 8],
        40,
    );
    assert_eq!(part.self_ref_mask, 0b0000_0011);
    assert_eq!(part.exterior_mask, 0b0011_0000);
    let mut tension = liminika_core::DistortionTension::new(true);
    tension.step_hot_path_branchless(40, part.boundary_sensitivity_q7, 32, 16, false);
    let spine = liminika_core::SystemSpineState::new();
    let ego = liminika_core::LiminalEgoState::from_preset(liminika_core::LiminalEgoPreset::Medium);
    let net = liminika_core::EgoDifferentialNet::from_ego_state_and_axes(
        &ego,
        &ego.extract_intent_axes(),
        3,
    );
    let stig = liminika_core::StigmergicGranuleState::new();
    let mut pframe_target = liminika_core::GatewayFrame::default();
    let _ = net.evaluate_pframe_chain_branchless(
        &ego,
        &spine,
        &part,
        &stig,
        &tension,
        &mut pframe_target,
    );
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
}
