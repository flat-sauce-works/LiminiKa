// name: tests/core/test_zero_alloc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Integration test enforcing zero dynamic heap allocation on the GCSO Hot Path.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use liminika_core::abi::{
    gcso_action_hub_resolve_trail_in_swarm, gcso_config_init_default, gcso_config_t,
    gcso_context_create, gcso_context_destroy, gcso_context_step_token,
    gcso_dpsr_apply_phase_steering_fused_safe, gcso_pointer_trail_t, GcsoContextHandle,
    GCSO_SUCCESS,
};
use liminika_core::traits::aligned::{AlignedBuffer16, AlignedBuffer32};
use liminika_core::traits::telemetry::StigmergicReconstitutionReport;
use liminika_core::traits::{DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD};

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
    let mut config = gcso_config_t::default();
    // SAFETY: `config` is a valid mutable stack reference.
    let status = unsafe { gcso_config_init_default(&mut config) };
    assert_eq!(status, GCSO_SUCCESS);

    let mut ctx_handle: GcsoContextHandle = std::ptr::null_mut();
    // SAFETY: `config` is initialized and `ctx_handle` is a valid out-pointer.
    let status = unsafe { gcso_context_create(&config, &mut ctx_handle) };
    assert_eq!(status, GCSO_SUCCESS);
    assert!(!ctx_handle.is_null());

    let mut query_buf = AlignedBuffer32::<f32, 128>::new([0.5; 128]);
    let mut key_buf = AlignedBuffer32::<f32, 128>::new([0.25; 128]);
    let mut trail_out = gcso_pointer_trail_t::default();

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
        // 128-element buffers, and `trail_out` is 128-byte aligned.
        let step_status = unsafe {
            gcso_context_step_token(
                ctx_handle,
                token_id,
                query_buf.as_mut_slice().as_mut_ptr(),
                key_buf.as_mut_slice().as_mut_ptr(),
                &mut trail_out,
            )
        };
        assert_eq!(step_status, GCSO_SUCCESS);

        // SAFETY: `steer_q` and `steer_k` are disjoint 32B-aligned tensors, `phase_buf` is 16B-aligned.
        let steer_status = unsafe {
            gcso_dpsr_apply_phase_steering_fused_safe(
                steer_q.as_mut_slice().as_mut_ptr(),
                steer_k.as_mut_slice().as_mut_ptr(),
                phase_buf.as_slice().as_ptr(),
                HEAD_DIM,
                NUM_HEADS,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
            )
        };
        assert_eq!(steer_status, GCSO_SUCCESS);

        // SAFETY: `trail_out`, `phase_buf`, and `swarm_report` are valid, aligned, and disjoint.
        let swarm_status = unsafe {
            gcso_action_hub_resolve_trail_in_swarm(
                &mut trail_out,
                u64::from(token_id) + 1,
                phase_buf.as_slice().as_ptr(),
                0xFF,
                64,
                ((DEFAULT_QDPS_MIN_STEP_RAD / core::f32::consts::PI) * 128.0) as u8,
                16,
                4,
                0.5,
                0.2,
                &mut swarm_report,
            )
        };
        assert_eq!(swarm_status, GCSO_SUCCESS);
    }

    // 4. Disable Guard before Cold Path cleanup.
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
    // SAFETY: `ctx_handle` was created by `gcso_context_create` and is destroyed once.
    let destroy_status = unsafe { gcso_context_destroy(ctx_handle) };
    assert_eq!(destroy_status, GCSO_SUCCESS);
}

#[test]
fn test_runtime_context_pierce_and_slide_in_zero_alloc() {
    use liminika_core::traits::gateway::{
        AutonomousCoreLayerStack, GatewayFrame, HorizonTriggerInput, LatentHorizonSlot,
        ObjectiveFact, SubjectivePersona, SystemConstraint,
    };
    use liminika_core::traits::rope::{RopeGeometryDescriptor, RopePairLayout, RopeSteeringTarget};
    use liminika_core::traits::EGO_TRIGGER_BOUNDARY_CHATTER;
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
            stack: &mut AutonomousCoreLayerStack,
            _trigger: &HorizonTriggerInput,
        ) {
            stack.liminal_layer.active_aperture_mask |= 1 << 5;
            frame.phase_deltas[0] = 16;
        }
    }

    let mut ctx = RuntimeContext::new(Config::new()).unwrap();
    let mut q_buf = AlignedBuffer32::<f32, { 128 * 32 }>::new([1.0_f32; 128 * 32]);
    let mut trail = gcso_pointer_trail_t::default();
    let signal = (
        SystemConstraint::new((), 0.2, 0.0, 0xFF),
        ObjectiveFact::new([10i8; 64], 1.2, 0x2000, 11),
        SubjectivePersona::new([0i8; 64], 0.0, 0, 0),
    );
    let mut slot = ZeroAllocSlideSlot;
    let mut no_esc = ();

    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(true));
    for step in 0..32 {
        ctx.core_layers_mut().ego_layer.last_intervened_mask = if (step & 1) == 0 {
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
    assert_eq!(snap.system.total_steps, 32);
    ENFORCE_ZERO_ALLOC.with(|flag| flag.set(false));
}
