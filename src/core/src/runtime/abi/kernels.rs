// name: src/core/src/runtime/abi/kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hot-path `float*`-only tensor kernel and Lacuna partition C-ABI entry points
//! (`v0.4.0`, `0x0000_0400`, Strict Provenance `.addr()` compliant).

use super::*;
use crate::runtime::gateway::{GatewayPhaseSteeringExt, GatewaySteerInput};

/// Applies RoPE phase steering to a 32B-aligned Query tensor (`float*`) in zero-allocation mode.
///
/// - Default (`flags == 0`) applies soft-bounded `PhaseClamp` (`max_rad`, or
///   [`DEFAULT_PHASE_CLAMP_MAX_RAD`] when `max_rad == 0.0`) dispatched across `Interleaved` or
///   `HalfSplit` partial-RoPE geometry.
/// - `LMNK_ROPE_STEER_FLAG_UNCLAMPED` applies direct full-head `Interleaved` rotation.
/// - `num_tokens != 1` fails closed with [`LMNK_ERR_NOT_IMPLEMENTED`].
///
/// # Safety
/// `params` and its non-null tensor pointers must satisfy alignment, bounds, and disjointness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_rope_steer_q(params: *const lmnk_rope_steer_params_t) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment of `params`.
        let p = unsafe { FfiHandleRef::from_raw(params)? };
        validate_param_struct_size::<lmnk_rope_steer_params_t>(p.struct_size)?;
        if (p.flags & !LMNK_ROPE_STEER_KNOWN_FLAGS) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (engine, geom, total_q_len) =
            init_rope_geometry(p.head_dim, p.rotary_dim, p.num_heads, p.rope_layout)?;
        validate_token_batch(p.num_tokens, p.token_stride_elems, total_q_len)?;
        validate_ptr_aligned(p.query_tensor, 32)?;
        validate_ptr_aligned(p.phase_deltas, 16)?;

        let q_bytes = validate_raw_slice_bounds(p.query_tensor, total_q_len)?;
        let p_bytes = validate_raw_slice_bounds(p.phase_deltas, geom.num_heads)?;
        check_disjoint_rw(
            &[(p.query_tensor.addr(), q_bytes)],
            &[
                (p.phase_deltas.addr(), p_bytes),
                (params.addr(), size_of::<lmnk_rope_steer_params_t>()),
            ],
        )?;

        // SAFETY: Validated for non-nullness, 32B/16B alignment, bounds, and disjointness.
        let (mut q_aligned, p_aligned) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut(p.query_tensor, total_q_len)?,
                AlignedSlice16::from_raw_parts(p.phase_deltas, geom.num_heads)?,
            )
        };

        if (p.flags & LMNK_ROPE_STEER_FLAG_UNCLAMPED) != 0 {
            if geom.layout != RopePairLayout::Interleaved || geom.rotary_dim != geom.head_dim {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            engine.apply_phase_steering(&mut q_aligned, &p_aligned)?;
        } else {
            if !branchless_is_finite_f32(p.max_rad) || p.max_rad < 0.0 {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            let clamp_rad = if p.max_rad == 0.0 {
                DEFAULT_PHASE_CLAMP_MAX_RAD
            } else {
                p.max_rad
            };
            PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
                &engine,
                &geom,
                &mut q_aligned,
                &p_aligned,
                clamp_rad,
            )?;
        }
        Ok(LMNK_OK)
    })
}

/// Recovers per-head Q0.7 Procrustes phase deltas from a `(q_before, q_after)` RoPE pair (`0 alloc`).
///
/// # Safety
/// `params` and its tensor pointers must satisfy alignment, bounds, and disjointness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_rope_procrustes_q7(
    params: *const lmnk_rope_procrustes_params_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment of `params`.
        let p = unsafe { FfiHandleRef::from_raw(params)? };
        validate_param_struct_size::<lmnk_rope_procrustes_params_t>(p.struct_size)?;
        if p.flags != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (engine, geom, total_q_len) =
            init_rope_geometry(p.head_dim, p.rotary_dim, p.num_heads, p.rope_layout)?;
        validate_token_batch(p.num_tokens, p.token_stride_elems, total_q_len)?;
        validate_ptr_aligned(p.q_before, 32)?;
        validate_ptr_aligned(p.q_after, 32)?;
        validate_ptr_aligned(p.q7_phase_out, 16)?;

        let qb_bytes = validate_raw_slice_bounds(p.q_before, total_q_len)?;
        let qa_bytes = validate_raw_slice_bounds(p.q_after, total_q_len)?;
        let out_bytes = validate_raw_slice_bounds(p.q7_phase_out, geom.num_heads)?;
        check_disjoint_rw(
            &[(p.q7_phase_out.addr(), out_bytes)],
            &[
                (p.q_before.addr(), qb_bytes),
                (p.q_after.addr(), qa_bytes),
                (params.addr(), size_of::<lmnk_rope_procrustes_params_t>()),
            ],
        )?;

        // SAFETY: Validated for non-nullness, 32B/16B alignment, bounds, and disjointness.
        let (q_bef_aligned, q_aft_aligned, mut p_out_aligned) = unsafe {
            (
                AlignedSlice32::from_raw_parts(p.q_before, total_q_len)?,
                AlignedSlice32::from_raw_parts(p.q_after, total_q_len)?,
                AlignedSliceMut16::from_raw_parts_mut(p.q7_phase_out, geom.num_heads)?,
            )
        };

        PhaseSteeringGeometryExt::compute_procrustes_phase_delta_for_geometry(
            &engine,
            &geom,
            &q_bef_aligned,
            &q_aft_aligned,
            &mut p_out_aligned,
        )?;
        Ok(LMNK_OK)
    })
}

/// Evaluates an arena-indexed 32-byte [`lmnk_rank1_descriptor_t`] residual kernel (`0 alloc`).
///
/// All descriptor vectors (`u_span`, `v_span`, optional `gain_offset`) are resolved with checked
/// bounds inside the caller-owned `arena` slice (100% Strict Provenance compliant).
///
/// # Safety
/// `params` and its pointers must satisfy alignment, bounds, and disjointness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_rank1_eval(params: *const lmnk_rank1_eval_params_t) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment of `params`.
        let p = unsafe { FfiHandleRef::from_raw(params)? };
        validate_param_struct_size::<lmnk_rank1_eval_params_t>(p.struct_size)?;
        if (p.flags & !LMNK_RANK1_KNOWN_FLAGS) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let dim_in = p.dim_in as usize;
        let dim_out = p.dim_out as usize;
        let arena_len = p.arena_len as usize;
        if dim_in == 0 || dim_out == 0 || arena_len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let max_dim = dim_in.max(dim_out);
        validate_token_batch(p.num_tokens, p.token_stride_elems, max_dim)?;

        validate_ptr(p.desc)?;
        validate_ptr(p.arena)?;
        validate_ptr_aligned(p.x_in, 32)?;
        validate_ptr_aligned(p.y_inout, 32)?;

        let arena_bytes = validate_raw_slice_bounds(p.arena, arena_len)?;
        let x_bytes = validate_raw_slice_bounds(p.x_in, dim_in)?;
        let y_bytes = validate_raw_slice_bounds(p.y_inout, dim_out)?;
        check_disjoint_rw(
            &[(p.y_inout.addr(), y_bytes)],
            &[
                (p.desc.addr(), size_of::<lmnk_rank1_descriptor_t>()),
                (p.arena.addr(), arena_bytes),
                (p.x_in.addr(), x_bytes),
                (params.addr(), size_of::<lmnk_rank1_eval_params_t>()),
            ],
        )?;

        // SAFETY: All pointers are validated non-null, aligned, bounded, and disjoint above.
        let (desc_ref, arena_slice, x_aligned, mut y_aligned) = unsafe {
            (
                FfiHandleRef::from_raw(p.desc)?,
                checked_slice_from_raw_parts(p.arena, arena_len)?,
                AlignedSlice32::from_raw_parts(p.x_in, dim_in)?,
                AlignedSliceMut32::from_raw_parts_mut(p.y_inout, dim_out)?,
            )
        };

        if (p.flags & LMNK_RANK1_FLAG_FUSED) != 0 {
            eval_rank1_fused(&mut y_aligned, &x_aligned, arena_slice, desc_ref.as_ref())?;
        } else {
            eval_rank1(&mut y_aligned, &x_aligned, arena_slice, desc_ref.as_ref())?;
        }
        Ok(LMNK_OK)
    })
}

/// Steers a 32B-aligned Query tensor using a 2-Origin `(ObjectiveFact, SubjectivePersona)`
/// Manifold Gateway composite (`0 alloc`) with optional [`lmnk_gateway_frame_t`] and
/// [`lmnk_causal_report_t`] telemetry outputs.
///
/// # Safety
/// `params` and its non-null pointers must satisfy alignment, bounds, and pairwise disjointness.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_gateway_steer(
    params: *const lmnk_gateway_steer_params_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and alignment of `params`.
        let p = unsafe { FfiHandleRef::from_raw(params)? };
        validate_param_struct_size::<lmnk_gateway_steer_params_t>(p.struct_size)?;
        if p.flags != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if !branchless_is_finite_f32(p.fact_energy) || !branchless_is_finite_f32(p.persona_energy) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let (engine, geom, total_q_len) =
            init_rope_geometry(p.head_dim, p.rotary_dim, p.num_heads, p.rope_layout)?;
        validate_token_batch(p.num_tokens, p.token_stride_elems, total_q_len)?;

        validate_ptr_aligned(p.query_tensor, 32)?;
        validate_ptr_aligned(p.fact_phases, 16)?;
        validate_ptr_aligned(p.persona_phases, 16)?;
        validate_ptr(p.trail_inout)?;

        let frame_span = if p.frame_out.is_null() {
            (0usize, 0usize)
        } else {
            validate_ptr(p.frame_out)?;
            (p.frame_out.addr(), size_of::<lmnk_gateway_frame_t>())
        };
        let report_span = if p.report_out.is_null() {
            (0usize, 0usize)
        } else {
            validate_ptr(p.report_out)?;
            (p.report_out.addr(), size_of::<lmnk_causal_report_t>())
        };

        let q_bytes = validate_raw_slice_bounds(p.query_tensor, total_q_len)?;
        let fp_bytes = validate_raw_slice_bounds(p.fact_phases, geom.num_heads)?;
        let pp_bytes = validate_raw_slice_bounds(p.persona_phases, geom.num_heads)?;
        check_disjoint_rw(
            &[
                (p.query_tensor.addr(), q_bytes),
                (p.trail_inout.addr(), size_of::<lmnk_pointer_trail_t>()),
                frame_span,
                report_span,
            ],
            &[
                (p.fact_phases.addr(), fp_bytes),
                (p.persona_phases.addr(), pp_bytes),
                (params.addr(), size_of::<lmnk_gateway_steer_params_t>()),
            ],
        )?;

        // SAFETY: All pointers are validated non-null (or optional), aligned, bounded, and disjoint.
        let (mut trail, mut q_aligned, fact_slice, persona_slice, frame_opt, report_opt) = unsafe {
            (
                FfiHandleMut::from_raw(p.trail_inout)?,
                AlignedSliceMut32::from_raw_parts_mut(p.query_tensor, total_q_len)?,
                AlignedSlice16::from_raw_parts(p.fact_phases, geom.num_heads)?,
                AlignedSlice16::from_raw_parts(p.persona_phases, geom.num_heads)?,
                FfiHandleMut::from_raw_opt(p.frame_out)?,
                FfiHandleMut::from_raw_opt(p.report_out)?,
            )
        };

        let mut fact_arr = [0i8; MAX_ATTENTION_HEADS];
        let mut persona_arr = [0i8; MAX_ATTENTION_HEADS];
        fact_arr[..geom.num_heads].copy_from_slice(fact_slice.as_slice());
        persona_arr[..geom.num_heads].copy_from_slice(persona_slice.as_slice());

        let composite = (
            ObjectiveFact::new(fact_arr, p.fact_energy, 0, 1),
            SubjectivePersona::new(persona_arr, p.persona_energy, 0, 2),
        );
        let damping = if p.unresolved_damping_u7 == 0 {
            128
        } else {
            p.unresolved_damping_u7
        };

        let (frame, report) =
            GatewayPhaseSteeringExt::steer_gateway_signal_with_core_layers_for_geometry_safe(
                &engine,
                GatewaySteerInput {
                    signal: &composite,
                    geometry: &geom,
                    core_ctx: None,
                    unresolved_damping_u7: damping,
                },
                &mut trail,
                &mut q_aligned,
                None,
                None,
            )?;
        if let Some(mut dst_frame) = frame_opt {
            *dst_frame = frame;
        }
        if let Some(mut dst_report) = report_opt {
            *dst_report = report;
        }
        Ok(LMNK_OK)
    })
}

/// Evaluates the 8-granule branchless 4-state Lacuna partition (`Unobserved`, `SelfRef`,
/// `Exterior`, `Undefined`) in zero-allocation mode (`0 alloc`, `0 fdiv`).
///
/// # Safety
/// `params` and `out` must be non-null, naturally aligned, and disjoint.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_lacuna_partition_eval(
    params: *const lmnk_lacuna_eval_params_t,
    out: *mut lmnk_lacuna_partition_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(params)?;
        validate_ptr(out)?;
        check_disjoint_rw(
            &[(out.addr(), size_of::<lmnk_lacuna_partition_t>())],
            &[(params.addr(), size_of::<lmnk_lacuna_eval_params_t>())],
        )?;
        // SAFETY: `params` and `out` are validated non-null, aligned, and disjoint above.
        let (p, mut out_ref) = unsafe {
            (
                FfiHandleRef::from_raw(params)?,
                FfiHandleMut::from_raw(out)?,
            )
        };
        validate_param_struct_size::<lmnk_lacuna_eval_params_t>(p.struct_size)?;

        *out_ref = evaluate_lacuna_partition_branchless(
            p.self_ref_mask,
            &p.other_density_q7,
            &p.noise_tolerance_q7,
            p.deadband_q7,
        );
        Ok(LMNK_OK)
    })
}

/// Evaluates one standalone Tier-1 safety-envelope constraint step (`0 alloc`, `0 fdiv`)
/// directly on a 64-byte [`lmnk_envelope_state_t`] without requiring a 5,120-byte `RuntimeContext`.
///
/// Caller must initialize `params->struct_size = sizeof(lmnk_envelope_step_params_t)` (`96`).
/// When `LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS` is set in `params->flags`, `params->bounds` is
/// ignored without validation and canonical [`EnvelopeBounds::new()`] is applied instead.
///
/// # Safety
/// `state_inout` (64B aligned), `params` (32B aligned), and `feedback_out` (16B aligned) must be
/// non-null, naturally aligned, and pairwise disjoint in memory.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lmnk_envelope_step(
    state_inout: *mut lmnk_envelope_state_t,
    params: *const lmnk_envelope_step_params_t,
    feedback_out: *mut lmnk_envelope_feedback_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(state_inout)?;
        validate_ptr(params)?;
        validate_ptr(feedback_out)?;
        check_disjoint_rw(
            &[
                (state_inout.addr(), size_of::<lmnk_envelope_state_t>()),
                (feedback_out.addr(), size_of::<lmnk_envelope_feedback_t>()),
            ],
            &[(params.addr(), size_of::<lmnk_envelope_step_params_t>())],
        )?;
        // SAFETY: `state_inout`, `params`, and `feedback_out` are verified non-null, aligned, and disjoint.
        let (mut state, p, mut out) = unsafe {
            (
                FfiHandleMut::from_raw(state_inout)?,
                FfiHandleRef::from_raw(params)?,
                FfiHandleMut::from_raw(feedback_out)?,
            )
        };
        validate_param_struct_size::<lmnk_envelope_step_params_t>(p.struct_size)?;
        if (p.flags & !LMNK_ENVELOPE_STEP_KNOWN_FLAGS) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if !state.validate_invariants() {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        if p.signal.pressure_q15 > SPINE_ENTROPY_Q15_ONE
            || p.signal.delta_q15 > SPINE_ENTROPY_Q15_ONE
            || p.signal.contraction_q7 > 128
            || p.signal.chatter_q7 > 128
            || (p.signal.signal_flags & !BOUNDARY_SIGNAL_KNOWN_FLAGS) != 0
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let eff_bounds = if (p.flags & LMNK_ENVELOPE_STEP_FLAG_DEFAULT_BOUNDS) != 0 {
            EnvelopeBounds::new()
        } else {
            if !p.bounds.validate_invariants() {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            p.bounds
        };

        *out = state.step(&eff_bounds, &p.signal);
        Ok(LMNK_OK)
    })
}
