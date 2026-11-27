// name: tests/core/test_zero_alloc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test enforcing zero dynamic heap allocation on the LiminiKa Hot Path.

#[cfg(not(all(miri, windows)))]
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use liminika_core::abi::{
    LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT, LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS, LMNK_OK,
    LMNK_POLICY_FLAG_COMPUTE_POOL, LMNK_POLICY_FLAG_SPINE, LmnkContextHandle, lmnk_abi_info,
    lmnk_abi_info_t, lmnk_boundary_emit_params_t, lmnk_boundary_signal_t, lmnk_config_init_default,
    lmnk_config_t, lmnk_container_find_section, lmnk_container_section_view_t,
    lmnk_container_verify, lmnk_container_verify_report_t, lmnk_container_write,
    lmnk_container_write_params_t, lmnk_context_begin_turn, lmnk_context_configure_policy,
    lmnk_context_create, lmnk_context_deinit_in_place, lmnk_context_destroy,
    lmnk_context_emit_boundary_signal, lmnk_context_get_state, lmnk_context_init_in_place,
    lmnk_context_reset, lmnk_context_step_decode, lmnk_decode_step_params_t,
    lmnk_decode_step_report_t, lmnk_envelope_feedback_t, lmnk_envelope_state_t, lmnk_envelope_step,
    lmnk_envelope_step_params_t, lmnk_gateway_steer, lmnk_gateway_steer_params_t,
    lmnk_lacuna_eval_params_t, lmnk_lacuna_partition_eval, lmnk_lacuna_partition_t,
    lmnk_pointer_trail_t, lmnk_rank1_eval, lmnk_rank1_eval_params_t, lmnk_rope_steer_params_t,
    lmnk_rope_steer_q, lmnk_runtime_policy_t, lmnk_runtime_state_view_t, lmnk_section_input_t,
    lmnk_spine_decode_input_t, lmnk_system_pressure_reading_t, lmnk_turn_begin_params_t,
};
use liminika_core::foundation::{
    AlignedBuffer16, AlignedBuffer32, ArenaSpan, DEFAULT_PHASE_CLAMP_MAX_RAD, Pod,
    RANK1_GAIN_UNUSED, Rank1Descriptor,
};

thread_local! {
    static ENFORCE_ZERO_ALLOC: Cell<bool> = const { Cell::new(false) };
}

// On Windows Miri, `std::alloc::System` stores an over-aligned allocation header before the
// returned pointer, triggering an upstream Stacked Borrows mismatch inside `std::sys::alloc::windows`
// when dropping `Box<MaybeUninit<RuntimeContext>>` (64B-aligned). On Linux Miri and all native
// targets, `ZeroAllocGuardAllocator` runs unconditionally.
#[cfg(not(all(miri, windows)))]
struct ZeroAllocGuardAllocator;

#[cfg(not(all(miri, windows)))]
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

#[cfg(not(all(miri, windows)))]
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

    let mut trail_out = lmnk_pointer_trail_t::default();

    const HEAD_DIM: usize = 128;
    const NUM_HEADS: usize = 16;
    let mut steer_q =
        AlignedBuffer32::<f32, { HEAD_DIM * NUM_HEADS }>::new([1.0; HEAD_DIM * NUM_HEADS]);
    let mut phases = [0i8; 64];
    for (i, slot) in phases.iter_mut().enumerate() {
        *slot = ((i as i8).wrapping_mul(7)).wrapping_sub(32);
    }
    let phase_buf = AlignedBuffer16::<i8, 64>::new(phases);
    let arena = [0.25f32; 64];
    let r1_desc = Rank1Descriptor {
        layer_idx: 0,
        rank: 1,
        u_span: ArenaSpan::new(0, 32),
        v_span: ArenaSpan::new(32, 32),
        gain_offset: RANK1_GAIN_UNUSED,
        scale_factor: 0.5,
    };
    let r1_x = AlignedBuffer32::<f32, 32>::new([1.0; 32]);
    let mut r1_y = AlignedBuffer32::<f32, 32>::new([0.0; 32]);
    let lac_params = lmnk_lacuna_eval_params_t {
        self_ref_mask: 0x0F,
        deadband_q7: 16,
        other_density_q7: [8, 8, 8, 8, 64, 64, 0, 0],
        noise_tolerance_q7: [4; 8],
        ..Default::default()
    };
    let mut lac_out = lmnk_lacuna_partition_t::default();

    // 2. Enable Strict Zero-Allocation Guard for Hot Path.
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));

    // 3. Execute Hot Path step loop (verified zero-allocation).
    let hot_iters: u32 = if cfg!(miri) { 4 } else { 100 };
    for _ in 0..hot_iters {
        let steer_p = lmnk_rope_steer_params_t {
            head_dim: HEAD_DIM as u32,
            rotary_dim: HEAD_DIM as u32,
            num_heads: NUM_HEADS as u32,
            rope_layout: 0,
            max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            query_tensor: steer_q.as_mut_slice().as_mut_ptr(),
            phase_deltas: phase_buf.as_slice().as_ptr(),
            ..Default::default()
        };
        // SAFETY: `steer_q` is 32B-aligned and `phase_buf` is 16B-aligned and disjoint.
        assert_eq!(unsafe { lmnk_rope_steer_q(&steer_p) }, LMNK_OK);

        let gw_p = lmnk_gateway_steer_params_t {
            head_dim: HEAD_DIM as u32,
            rotary_dim: HEAD_DIM as u32,
            num_heads: NUM_HEADS as u32,
            rope_layout: 0,
            fact_energy: 1.5,
            persona_energy: 1.0,
            query_tensor: steer_q.as_mut_slice().as_mut_ptr(),
            fact_phases: phase_buf.as_slice().as_ptr(),
            persona_phases: phase_buf.as_slice().as_ptr(),
            trail_inout: &mut trail_out,
            ..Default::default()
        };
        // SAFETY: Buffers are valid, aligned, and disjoint.
        assert_eq!(unsafe { lmnk_gateway_steer(&gw_p) }, LMNK_OK);

        let r1_p = lmnk_rank1_eval_params_t {
            dim_in: 32,
            dim_out: 32,
            arena_len: arena.len() as u32,
            desc: &r1_desc,
            arena: arena.as_ptr(),
            x_in: r1_x.as_slice().as_ptr(),
            y_inout: r1_y.as_mut_slice().as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: Aligned disjoint buffers matching `dim_in == dim_out == 32`.
        assert_eq!(unsafe { lmnk_rank1_eval(&r1_p) }, LMNK_OK);

        // SAFETY: Valid stack pointers for `lac_params` and `lac_out`.
        let lac_status = unsafe { lmnk_lacuna_partition_eval(&lac_params, &mut lac_out) };
        assert_eq!(lac_status, LMNK_OK);
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

    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));

    // SAFETY: Stack backing satisfies `info.context_size` and 64-byte alignment without heap ops.
    unsafe {
        let mut info = lmnk_abi_info_t::default();
        assert_eq!(lmnk_abi_info(&mut info), LMNK_OK);
        let req_size = info.context_size as usize;
        let req_align = info.context_align as usize;
        assert_eq!(req_align, 64);
        assert!(req_size > 0 && req_size <= backing.0.len());
        assert_eq!(lmnk_config_init_default(&mut config), LMNK_OK);
        assert_eq!(
            lmnk_context_init_in_place(&config, backing.0.as_mut_ptr(), req_size),
            LMNK_OK
        );
        let ctx_handle = backing
            .0
            .as_mut_ptr()
            .cast::<liminika_core::abi::LmnkContextOpaque>();

        let pol = lmnk_runtime_policy_t {
            flags: LMNK_POLICY_FLAG_SPINE | LMNK_POLICY_FLAG_COMPUTE_POOL,
            baseline_pinned_compute_mask: 0x0081,
            active_compute_mask: 0x0000_FFFF,
            repulsion_compute_mask: 0x0000_0004,
            reversible_compute_mask: 0x0000_FFFC,
            baseline_pressure_q15: 8000,
            guard_hold_tokens: 3,
            max_observe_stride: 4,
            pre_bypass_enabled: 1,
            ..Default::default()
        };
        assert_eq!(lmnk_context_configure_policy(ctx_handle, &pol), LMNK_OK);

        let mut state_view = lmnk_runtime_state_view_t::default();
        assert_eq!(lmnk_context_get_state(ctx_handle, &mut state_view), LMNK_OK);
        assert_eq!(state_view.guard_hold_tokens, 3);
        assert_eq!(state_view.max_observe_stride, 4);
        assert_eq!(state_view.active_compute_mask, 0x0000_FFFF);

        let turn = lmnk_turn_begin_params_t::default();
        assert_eq!(lmnk_context_begin_turn(ctx_handle, &turn), LMNK_OK);

        let decode_in = lmnk_decode_step_params_t {
            decode_input: lmnk_spine_decode_input_t::from_observed_entropy_f32(
                lmnk_system_pressure_reading_t::default(),
                0,
                0.35,
                8,
                false,
                true,
            )
            .with_exit_reward_override_q7(88),
            ..Default::default()
        };
        let mut decode_rep = lmnk_decode_step_report_t::default();
        assert_eq!(
            lmnk_context_step_decode(ctx_handle, &decode_in, &mut decode_rep),
            LMNK_OK
        );
        let emit_in = lmnk_boundary_emit_params_t {
            flags: LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT,
            pressure_q15: 10_000,
            chatter_q7: 8,
            epoch_id: 1,
            ..Default::default()
        };
        let mut sig_out = lmnk_boundary_signal_t::new();
        assert_eq!(
            lmnk_context_emit_boundary_signal(ctx_handle, &emit_in, &mut sig_out),
            LMNK_OK
        );
        let mut env_state = lmnk_envelope_state_t::new();
        let env_params = lmnk_envelope_step_params_t {
            flags: LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS,
            signal: sig_out,
            ..Default::default()
        };
        let mut env_fb = lmnk_envelope_feedback_t::default();
        assert_eq!(
            lmnk_envelope_step(&mut env_state, &env_params, &mut env_fb),
            LMNK_OK
        );
        assert_eq!(env_fb.contraction_q7, 128);
        assert_eq!(lmnk_context_reset(ctx_handle), LMNK_OK);
        assert_eq!(lmnk_context_deinit_in_place(ctx_handle), LMNK_OK);
        assert!(backing.0[..req_size].iter().all(|&b| b == 0));
        assert_eq!(
            lmnk_context_deinit_in_place(ctx_handle),
            liminika_core::abi::LMNK_ERR_INVALID_STATE
        );

        let anchor = liminika_core::PhaseAnchorRecord::new_empty(
            7,
            liminika_core::stigmergy::ATTRACTOR_SLOT_PERSONA,
            0x1000,
        );
        let anchor_bytes = anchor.as_bytes();
        let sec = lmnk_section_input_t {
            kind: liminika_core::LMNK_SECTION_PHASE_ANCHORS,
            flags: 0,
            payload_len: anchor_bytes.len() as u32,
            payload: anchor_bytes.as_ptr(),
            ..Default::default()
        };
        #[repr(C, align(64))]
        struct AlignedContainerBacking([u8; 512]);
        let mut container_buf = AlignedContainerBacking([0u8; 512]);
        let mut written = 0usize;
        let wp = lmnk_container_write_params_t {
            container_flags: liminika_core::LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            timestamp_epoch_sec: 1_774_900_000,
            section_count: 1,
            sections: &sec,
            out_bytes: container_buf.0.as_mut_ptr(),
            out_capacity: container_buf.0.len(),
            written_bytes_out: &mut written,
            ..Default::default()
        };
        assert_eq!(lmnk_container_write(&wp), LMNK_OK);
        let mut vrep = lmnk_container_verify_report_t::default();
        assert_eq!(
            lmnk_container_verify(container_buf.0.as_ptr(), written, &mut vrep),
            LMNK_OK
        );
        assert_eq!(vrep.section_count, 1);
        let mut sview = lmnk_container_section_view_t::default();
        assert_eq!(
            lmnk_container_find_section(
                container_buf.0.as_ptr(),
                written,
                liminika_core::LMNK_SECTION_PHASE_ANCHORS,
                &mut sview,
            ),
            LMNK_OK
        );
        assert_eq!(sview.length as usize, anchor_bytes.len());
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

    // Verify label-free AnonymousComputePool dispatch, whole-token bypass, LacunaPartition8,
    // Envelope isolation, DistortionDelta P-Frame chain, and DistortionTension remain zero-alloc.
    ctx.pin_compute_slots(0b0001);
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
    let _ = ctx.evaluate_whole_token_bypass_with_spine(
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
        false,
        &mut dispatch,
    );
    ctx.begin_turn_spine_policy(false, false);
    let pressure = liminika_core::drivers::SystemPressureReading::default();
    let _ = ctx.step_decode_control(
        &liminika_core::drivers::SpineDecodeInput::from_observed_entropy_f32(
            pressure, 0, 0.35, 8, false, true,
        ),
        false,
    );
    let _ = ctx.step_decode_control(
        &liminika_core::drivers::SpineDecodeInput::unobserved(pressure, 1, 8, false, true),
        false,
    );
    let _ = ctx.step_decode_control(
        &liminika_core::drivers::SpineDecodeInput::from_observed_entropy_f32(
            pressure, 2, 0.90, 96, false, true,
        ),
        false,
    );
    let _ = ctx.step_decode_control(
        &liminika_core::drivers::SpineDecodeInput::from_observed_entropy_f32(
            pressure,
            3,
            f32::NAN,
            8,
            false,
            true,
        ),
        false,
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
    let envelope = liminika_core::EnvelopeState::new();
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
        &envelope,
        &part,
        &stig,
        &tension,
        &mut pframe_target,
    );
    let solo_guard = liminika_core::runtime::GuardPhaseAnchor::new([i8::MIN; 64], 0.85, 0.50);
    let solo_rep_frame =
        liminika_core::runtime::ManifoldGatewaySignal::synthesize_gateway_frame(&solo_guard);
    assert!(solo_rep_frame.is_repulsed());
    assert_eq!(solo_rep_frame.phase_deltas, [127i8; 64]);

    let mut attract_arr = [0i8; 64];
    attract_arr[..32].fill(40);
    let mut repel_arr = [0i8; 64];
    repel_arr[32..].fill(32);
    let sys_sig = liminika_core::runtime::SystemConstraint::new((), 0.0, 0.0, 0xFF);
    let attract_sig = liminika_core::runtime::ObjectiveFact::new(attract_arr, 2.0, 0, 7);
    let repel_sig = liminika_core::runtime::GuardPhaseAnchor::new(repel_arr, 0.85, 0.50);
    let combo_frame = liminika_core::runtime::synthesize_multi_origin_gateway_frame(
        Some(&sys_sig),
        Some(&attract_sig),
        None::<&liminika_core::runtime::SubjectivePersona<[i8; 64]>>,
        Some(&repel_sig),
    );
    assert!(combo_frame.is_repulsed());
    assert_eq!(combo_frame.phase_deltas[0], 40);
    assert_eq!(combo_frame.phase_deltas[32], -32);
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
}
