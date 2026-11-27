// name: src/core/src/runtime/abi/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C-ABI interface definitions, FFI safety barriers, versioned parameter struct layouts, and the
//! 27-function C11/C++20 public surface for `LiminiKa` `v0.4.0` (`0x0000_0400`).

// C-ABI typedef aliases keep their `lmnk_*_t` spelling to mirror `include/liminika/lmnk_types.h`.
#![allow(non_camel_case_types)]

#[cfg(feature = "std")]
extern crate std;

use alloc::boxed::Box;
use core::ffi::c_char;
use core::mem::{align_of, offset_of, size_of};
use core::ptr;
use core::slice;

use crate::foundation::{
    AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32,
    DEFAULT_PHASE_CLAMP_MAX_RAD, PhaseSteering, PhaseSteeringGeometryExt, RopeGeometryDescriptor,
    RopePairLayout, branchless_is_finite_f32, eval_rank1, eval_rank1_fused, is_ptr_aligned_to,
    raw_byte_ranges_overlap, validate_raw_slice_bounds,
};
use crate::liminal::{LmnkResult, MAX_ATTENTION_HEADS, evaluate_lacuna_partition_branchless};
use crate::runtime::context::{
    CTX_ALLOC_HEAP_BOXED, CTX_ALLOC_IN_PLACE, Config, RuntimeContext, init_default_config,
    query_capability, validate_config_invariants,
};
use crate::runtime::gateway::{ObjectiveFact, SubjectivePersona};
use crate::stigmergy::PhaseSteeringEngine;

/// Helper macro to catch unwinding panics and safely bridge execution into C-ABI status codes.
#[macro_export]
macro_rules! ffi_boundary {
    ($body:expr) => {{
        let run_ffi_body =
            || -> Result<$crate::runtime::abi::LmnkStatus, $crate::runtime::abi::LmnkStatus> {
                $body
            };

        #[cfg(feature = "std")]
        let status = match ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(run_ffi_body))
        {
            Ok(Ok(status)) => status,
            Ok(Err(err_status)) => err_status,
            Err(_) => $crate::runtime::abi::LMNK_ERR_PANIC_CAUGHT,
        };

        #[cfg(not(feature = "std"))]
        let status = match run_ffi_body() {
            Ok(status) => status,
            Err(err_status) => err_status,
        };

        status
    }};
}

pub mod handles;
pub use handles::*;

pub mod types;
pub use types::*;

mod dispatch;
pub use dispatch::is_aligned;
use dispatch::*;

pub mod runtime;
pub use runtime::*;

pub mod kernels;
pub use kernels::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::pod::Pod;
    use crate::foundation::{AlignedBuffer16, AlignedBuffer32, AlignedBuffer64};
    use core::ffi::CStr;

    #[test]
    fn test_v040_handshake_and_config_guards() {
        assert_eq!(lmnk_abi_version_hex(), 0x0000_0400);
        // SAFETY: `lmnk_abi_version_string` and `lmnk_status_to_string` return static C strings.
        unsafe {
            assert_eq!(
                CStr::from_ptr(lmnk_abi_version_string()).to_str().unwrap(),
                "0.4.0"
            );
            assert_eq!(
                CStr::from_ptr(lmnk_status_to_string(LMNK_OK))
                    .to_str()
                    .unwrap(),
                "LMNK_OK"
            );
        }

        let mut bad_info = lmnk_abi_info_t {
            struct_size: 0,
            ..Default::default()
        };
        // SAFETY: Valid stack pointer.
        unsafe {
            assert_eq!(lmnk_abi_info(&mut bad_info), LMNK_ERR_INVALID_ARGUMENT);
        }

        let mut info = lmnk_abi_info_t::default();
        // SAFETY: Valid stack pointer with `struct_size == size_of::<lmnk_abi_info_t>()`.
        unsafe {
            assert_eq!(lmnk_abi_info(&mut info), LMNK_OK);
        }
        assert_eq!(info.abi_version_hex, 0x0000_0400);
        assert_eq!(info.context_size as usize, size_of::<RuntimeContext>());
        assert_eq!(info.context_align as usize, align_of::<RuntimeContext>());
        assert_eq!(
            info.snapshot_max_bytes as usize,
            LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES
        );
        assert_eq!(info.topology_crc32, CORE_TOPOLOGY.topology_crc32);
        assert_ne!(info.capability_flags & LMNK_CAP_SWAR_BASELINE, 0);

        let mut cfg = lmnk_config_t::default();
        // SAFETY: Valid 64B-aligned stack struct.
        unsafe {
            assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
            assert_eq!(cfg.struct_size, 64);
            assert_eq!(lmnk_config_validate(&cfg), LMNK_OK);
            cfg.struct_size = 0;
            assert_eq!(lmnk_config_validate(&cfg), LMNK_ERR_INVALID_ARGUMENT);
            cfg.struct_size = u32::MAX;
            assert_eq!(lmnk_config_validate(&cfg), LMNK_ERR_INVALID_ARGUMENT);
        }
    }

    #[test]
    fn test_v040_context_lifecycle_policy_decode_and_snapshot() {
        let mut cfg = lmnk_config_t::default();
        let mut ctx: LmnkContextHandle = ptr::null_mut();
        // SAFETY: Stack pointers are valid and aligned.
        unsafe {
            assert_eq!(lmnk_config_init_default(&mut cfg), LMNK_OK);
            assert_eq!(lmnk_context_create(&cfg, &mut ctx), LMNK_OK);
            assert!(!ctx.is_null());

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
            assert_eq!(lmnk_context_configure_policy(ctx, &pol), LMNK_OK);

            let turn = lmnk_turn_begin_params_t::default();
            assert_eq!(lmnk_context_begin_turn(ctx, &turn), LMNK_OK);

            let step_in = lmnk_decode_step_params_t::default();
            let mut step_out = lmnk_decode_step_report_t::default();
            assert_eq!(
                lmnk_context_step_decode(ctx, &step_in, &mut step_out),
                LMNK_OK
            );
            assert_eq!(step_out.dispatch.execute_mask & 0x0081, 0x0081);

            let mut view = lmnk_runtime_state_view_t::default();
            assert_eq!(lmnk_context_get_state(ctx, &mut view), LMNK_OK);
            assert_eq!(view.baseline_pinned_compute_mask & 0x0081, 0x0081);
            assert_eq!(view.active_compute_mask, 0x0000_FFFF);
            assert_eq!(view.guard_hold_tokens, 3);
            assert_eq!(view.max_observe_stride, 4);
            assert_eq!(view.envelope_contraction_q7, 128);
            assert_eq!(view.envelope_breach_mask, ENVELOPE_BREACH_NONE);

            let emit_in = lmnk_boundary_emit_params_t {
                flags: LMNK_BOUNDARY_EMIT_FLAG_IRREVERSIBLE_INTENT,
                pressure_q15: 9_600,
                chatter_q7: 12,
                epoch_id: 5,
                ..Default::default()
            };
            let mut sig_out = lmnk_boundary_signal_t::new();
            assert_eq!(
                lmnk_context_emit_boundary_signal(ctx, &emit_in, &mut sig_out),
                LMNK_OK
            );
            assert_eq!(sig_out.pressure_q15, 9_600);
            assert_eq!(sig_out.chatter_q7, 12);
            assert_eq!(sig_out.epoch_id, 5);
            assert_ne!(
                sig_out.signal_flags & BOUNDARY_SIGNAL_FLAG_IRREVERSIBLE_INTENT,
                0
            );

            // Tier-1 standalone envelope step with DEFAULT_BOUNDS ignores zeroed bounds.
            let mut env_state = lmnk_envelope_state_t::new();
            let env_params = lmnk_envelope_step_params_t {
                flags: LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS,
                bounds: lmnk_envelope_bounds_t {
                    immutable_compute_pin_mask: 0,
                    pressure_ceiling_q15: 0,
                    pressure_floor_q15: 60_000,
                    min_baseline_pressure_q15: 60_000,
                    max_baseline_pressure_q15: 0,
                    immutable_aperture_pin_mask: 0,
                    immutable_slot_pin_mask: 0,
                    immutable_action_pin_mask: 0,
                    max_chatter_q7: 255,
                    contraction_gain_q7: 0,
                    recovery_gain_q7: 0,
                    min_aperture_q7: 255,
                    ema_resume_dwell_target: 0,
                    reserved: [0; 8],
                },
                signal: sig_out,
                ..Default::default()
            };
            let mut env_fb = lmnk_envelope_feedback_t::default();
            assert_eq!(
                lmnk_envelope_step(&mut env_state, &env_params, &mut env_fb),
                LMNK_OK
            );
            assert_eq!(env_fb.contraction_q7, 128);
            assert_eq!(env_fb.breach_mask, ENVELOPE_BREACH_NONE);
            assert_ne!(
                env_state.baseline_pinned_compute_mask & HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
                0
            );

            let mut snap_len = 0usize;
            assert_eq!(
                lmnk_context_serialize(ctx, ptr::null_mut(), &mut snap_len),
                LMNK_OK
            );
            assert_eq!(snap_len, LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES);
            let mut buf = alloc::vec![0u8; snap_len];
            assert_eq!(
                lmnk_context_serialize(ctx, buf.as_mut_ptr(), &mut snap_len),
                LMNK_OK
            );

            let mut restored: LmnkContextHandle = ptr::null_mut();
            assert_eq!(
                lmnk_context_deserialize(buf.as_ptr(), snap_len, &mut restored),
                LMNK_OK
            );
            assert!(!restored.is_null());
            assert_eq!(lmnk_context_reset(restored), LMNK_OK);
            assert_eq!(lmnk_context_destroy(restored), LMNK_OK);
            assert_eq!(lmnk_context_destroy(ctx), LMNK_OK);
        }
    }

    #[test]
    fn test_v040_kernels_and_generic_container() {
        // 1. RoPE steer + Procrustes roundtrip + num_tokens != 1 fail-closed
        let mut q = AlignedBuffer32::new([1.0f32, 0.25, 0.5, -0.5, 1.0, 0.25, 0.5, -0.5]);
        let q_orig = q;
        let deltas = AlignedBuffer16::new([12i8, -12i8]);
        let mut recovered = AlignedBuffer16::new([0i8; 2]);

        let steer_p = lmnk_rope_steer_params_t {
            head_dim: 4,
            rotary_dim: 4,
            num_heads: 2,
            rope_layout: 1,
            max_rad: 0.12,
            query_tensor: q.as_mut_slice().as_mut_ptr(),
            phase_deltas: deltas.as_slice().as_ptr(),
            ..Default::default()
        };
        // SAFETY: Buffers are aligned, disjoint, and sized for `4 * 2`.
        unsafe {
            assert_eq!(lmnk_rope_steer_q(&steer_p), LMNK_OK);
            let mut bad_batch = steer_p;
            bad_batch.num_tokens = 2;
            assert_eq!(lmnk_rope_steer_q(&bad_batch), LMNK_ERR_NOT_IMPLEMENTED);

            let proc_p = lmnk_rope_procrustes_params_t {
                head_dim: 4,
                rotary_dim: 4,
                num_heads: 2,
                rope_layout: 1,
                q_before: q_orig.as_slice().as_ptr(),
                q_after: q.as_slice().as_ptr(),
                q7_phase_out: recovered.as_mut_slice().as_mut_ptr(),
                ..Default::default()
            };
            assert_eq!(lmnk_rope_procrustes_q7(&proc_p), LMNK_OK);
        }
        assert!(recovered.as_slice()[0] > 0);
        assert!(recovered.as_slice()[1] < 0);

        // 2. Arena-indexed Rank-1 evaluation + overflow fail-closed
        let arena = [2.0f32, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        let desc = Rank1Descriptor {
            layer_idx: 0,
            rank: 1,
            u_span: ArenaSpan::new(0, 4),
            v_span: ArenaSpan::new(4, 4),
            gain_offset: RANK1_GAIN_UNUSED,
            scale_factor: 1.0,
        };
        let x = AlignedBuffer32::new([1.0f32, 2.0, 3.0, 4.0]);
        let mut y = AlignedBuffer32::new([0.0f32; 4]);
        let r1_p = lmnk_rank1_eval_params_t {
            dim_in: 4,
            dim_out: 4,
            arena_len: arena.len() as u32,
            desc: &desc,
            arena: arena.as_ptr(),
            x_in: x.as_slice().as_ptr(),
            y_inout: y.as_mut_slice().as_mut_ptr(),
            ..Default::default()
        };
        // SAFETY: Aligned, disjoint buffers matching `dim_in == dim_out == 4`.
        unsafe {
            assert_eq!(lmnk_rank1_eval(&r1_p), LMNK_OK);
        }
        for &v in y.as_slice() {
            assert!((v - 20.0).abs() < 1e-4);
        }

        // 3. Gateway steer + Lacuna eval
        let mut trail = AlignedBuffer64::new([lmnk_pointer_trail_t::default()]);
        let mut frame_out = lmnk_gateway_frame_t::default();
        let mut report_out = lmnk_causal_report_t::default();
        let gw_p = lmnk_gateway_steer_params_t {
            head_dim: 4,
            rotary_dim: 4,
            num_heads: 2,
            rope_layout: 0,
            fact_energy: 2.0,
            persona_energy: 1.5,
            query_tensor: q.as_mut_slice().as_mut_ptr(),
            fact_phases: deltas.as_slice().as_ptr(),
            persona_phases: deltas.as_slice().as_ptr(),
            trail_inout: trail.as_mut_slice().as_mut_ptr(),
            frame_out: &mut frame_out,
            report_out: &mut report_out,
            ..Default::default()
        };
        let lac_p = lmnk_lacuna_eval_params_t {
            self_ref_mask: 0b0000_0011,
            deadband_q7: 8,
            other_density_q7: [80, 72, 60, 0, 0, 0, 0, 0],
            noise_tolerance_q7: [8; 8],
            ..Default::default()
        };
        let mut lac_out = lmnk_lacuna_partition_t::default();
        // SAFETY: Valid aligned stack/buffer pointers.
        unsafe {
            assert_eq!(lmnk_gateway_steer(&gw_p), LMNK_OK);
            assert_eq!(lmnk_lacuna_partition_eval(&lac_p, &mut lac_out), LMNK_OK);
        }
        assert_eq!(trail.as_slice()[0].step_count, 1);
        assert_ne!(lac_out.exterior_mask & (1 << 2), 0);

        // 4. Generic container write / verify / find_section
        let anchor = PhaseAnchorRecord::new_empty(7, ATTRACTOR_SLOT_PERSONA, 0);
        let anchor_bytes = anchor.as_bytes();
        let sec = lmnk_section_input_t {
            kind: LMNK_SECTION_PHASE_ANCHORS,
            flags: 0,
            payload_len: anchor_bytes.len() as u32,
            payload: anchor_bytes.as_ptr(),
            ..Default::default()
        };
        let mut req_len = 0usize;
        let query_p = lmnk_container_write_params_t {
            container_flags: LMNK_CONTAINER_FLAG_COMPILED_CACHE,
            timestamp_epoch_sec: 1_774_900_000,
            section_count: 1,
            sections: &sec,
            out_bytes: ptr::null_mut(),
            out_capacity: 0,
            written_bytes_out: &mut req_len,
            ..Default::default()
        };
        // SAFETY: Valid stack descriptors.
        unsafe {
            assert_eq!(lmnk_container_write(&query_p), LMNK_OK);
            assert!(req_len >= LMNK_CONTAINER_HEADER_BYTES + anchor_bytes.len());

            let mut container = alloc::vec![0u8; req_len];
            let mut written = 0usize;
            let write_p = lmnk_container_write_params_t {
                out_bytes: container.as_mut_ptr(),
                out_capacity: container.len(),
                written_bytes_out: &mut written,
                ..query_p
            };
            assert_eq!(lmnk_container_write(&write_p), LMNK_OK);
            assert_eq!(written, req_len);

            let mut verify_rep = lmnk_container_verify_report_t::default();
            assert_eq!(
                lmnk_container_verify(container.as_ptr(), written, &mut verify_rep),
                LMNK_OK
            );
            assert_eq!(verify_rep.abi_version, 0x0000_0400);
            assert_eq!(verify_rep.section_count, 1);

            let mut sec_view = lmnk_container_section_view_t::default();
            assert_eq!(
                lmnk_container_find_section(
                    container.as_ptr(),
                    written,
                    LMNK_SECTION_PHASE_ANCHORS,
                    &mut sec_view,
                ),
                LMNK_OK
            );
            assert_eq!(sec_view.length as usize, anchor_bytes.len());
            assert!(!sec_view.payload_ptr.is_null());

            let mut missing_view = lmnk_container_section_view_t::default();
            assert_eq!(
                lmnk_container_find_section(
                    container.as_ptr(),
                    written,
                    LMNK_SECTION_BLOOM_FILTER,
                    &mut missing_view,
                ),
                LMNK_ERR_ANCHOR_NOT_FOUND
            );
        }
    }
}
