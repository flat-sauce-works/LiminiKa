// name: src/core/src/runtime/abi/phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase Steering (PhaseSteering), PhaseStepFilter, 2-Origin Gateway, sub-head router routing, and Residual Adapter (Rank-1 residual) FFI.

use super::*;
use crate::runtime::gateway::GatewayPhaseSteeringExt;

// ===================================================================
// 4. PhaseSteering, PhaseStepFilter, 2-Origin Gateway, sub-head router & Rank-1 residual Interface
// ===================================================================

/// Creates a PhaseSteering phase steering kernel instance.
///
/// # Safety
/// `kernel_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_create(
    head_dim: u32,
    num_heads: u32,
    kernel_out: *mut LmnkPhaseSteeringHandle,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(kernel_out)? };
        let engine = PhaseSteeringEngine::new(head_dim, num_heads)?;

        *out_handle = Box::into_raw(Box::new(engine)).cast::<LmnkPhaseSteeringOpaque>();
        Ok(LMNK_OK)
    })
}

/// Applies inline PhaseSteering phase steering to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply(
    query_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering(&mut q_aligned, &p_aligned)?;
        Ok(LMNK_OK)
    })
}

/// Applies fused inline PhaseSteering phase steering simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_fused(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_disjoint_pair(&mut qk_pair, &p_aligned)?;
        Ok(LMNK_OK)
    })
}

/// Applies PhaseClamp soft-bounded $\tanh$ clamping ($\Delta\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\Delta\theta/\theta_{\max})$)
/// restricted to low-frequency RoPE channels (upper $d_{\mathrm{head}}/4$ dimensions).
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_safe(
    query_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(LMNK_OK)
    })
}

/// Applies PhaseClamp soft-bounded phase steering dispatched by [`lmnk_config_t`] RoPE geometry
/// (`Interleaved` vs `HalfSplit`, partial `rotary_dim`, `head_dim`, `num_heads`, `phase_clamp_max_rad`).
///
/// # Safety
/// `query_tensor` (32B aligned), `phase_deltas` (16B aligned), and `config` (16B aligned)
/// must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_with_config_safe(
    query_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    config: *const lmnk_config_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and 16-byte alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };
        let head_dim = cfg.head_dim as usize;
        let num_heads = cfg.num_heads as usize;
        let rotary_dim = if cfg.rotary_dim == 0 {
            head_dim
        } else {
            cfg.rotary_dim as usize
        };
        let layout = RopePairLayout::from_u8(cfg.rope_layout)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;
        let geom = RopeGeometryDescriptor {
            layout,
            head_dim,
            rotary_dim,
            num_heads,
        };

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
            &engine,
            &geom,
            &mut q_aligned,
            &p_aligned,
            cfg.phase_clamp_max_rad,
        )?;
        Ok(LMNK_OK)
    })
}

/// Steers `query_tensor` using a 2-Origin [`ManifoldGatewaySignal`]
/// composite (`(ObjectiveFact, SubjectivePersona)`) so structural hyperparameters are synthesized
/// autonomously from Gateway physics and dispatched via [`lmnk_config_t`] RoPE geometry.
///
/// # Safety
/// `query_tensor` (32B aligned), `fact_phases` (16B aligned), `persona_phases` (16B aligned),
/// `trail_inout` (128B aligned), and `config` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_steer_with_gateway_and_config_safe(
    query_tensor: *mut f32,
    fact_phases: *const lmnk_q7_t,
    fact_energy: f32,
    persona_phases: *const lmnk_q7_t,
    persona_energy: f32,
    trail_inout: *mut lmnk_pointer_trail_t,
    config: *const lmnk_config_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(fact_phases)?;
        validate_ptr(persona_phases)?;
        // SAFETY: `FfiHandleMut::from_raw` validates non-nullness and 128-byte alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
        // SAFETY: `FfiHandleRef::from_raw` validates non-nullness and 16-byte alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(config)? };

        let head_dim = cfg.head_dim as usize;
        let num_heads = cfg.num_heads as usize;
        let rotary_dim = if cfg.rotary_dim == 0 {
            head_dim
        } else {
            cfg.rotary_dim as usize
        };
        let layout = RopePairLayout::from_u8(cfg.rope_layout)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;
        let geom = RopeGeometryDescriptor {
            layout,
            head_dim,
            rotary_dim,
            num_heads,
        };

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let fact_slice = unsafe { AlignedSlice16::from_raw_parts(fact_phases, num_heads)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let persona_slice = unsafe { AlignedSlice16::from_raw_parts(persona_phases, num_heads)? };

        let mut fact_arr = [0i8; MAX_ATTENTION_HEADS];
        let mut persona_arr = [0i8; MAX_ATTENTION_HEADS];
        fact_arr[..num_heads].copy_from_slice(fact_slice.as_slice());
        persona_arr[..num_heads].copy_from_slice(persona_slice.as_slice());

        let composite = (
            ObjectiveFact::new(fact_arr, fact_energy, 0, 1),
            SubjectivePersona::new(persona_arr, persona_energy, 0, 2),
        );

        GatewayPhaseSteeringExt::steer_gateway_signal_for_geometry_safe(
            &engine,
            &composite,
            &geom,
            &mut trail,
            128,
            &mut q_aligned,
            None,
            None,
        )?;
        Ok(LMNK_OK)
    })
}

/// Recovers per-head Q7 Procrustes phase deltas from `HalfSplit` RoPE tensor pairs.
///
/// # Safety
/// `q_before` (32B aligned), `q_after` (32B aligned), and `q7_phase_out` (16B aligned)
/// must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_compute_procrustes_half_split(
    q_before: *const f32,
    q_after: *const f32,
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
    q7_phase_out: *mut lmnk_q7_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(q_before, 32)?;
        validate_ptr_aligned(q_after, 32)?;
        validate_ptr(q7_phase_out)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;
        let geom = RopeGeometryDescriptor {
            layout: RopePairLayout::HalfSplit,
            head_dim,
            rotary_dim,
            num_heads,
        };

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let q_bef_aligned = unsafe { AlignedSlice32::from_raw_parts(q_before, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let q_aft_aligned = unsafe { AlignedSlice32::from_raw_parts(q_after, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut p_out_aligned =
            unsafe { AlignedSliceMut16::from_raw_parts_mut(q7_phase_out, num_heads)? };

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

/// Applies fused PhaseClamp soft-bounded $\tanh$ clamping simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_fused_safe(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_fused_safe(
            &mut qk_pair.first,
            &mut qk_pair.second,
            &p_aligned,
            max_rad,
        )?;
        Ok(LMNK_OK)
    })
}

/// Applies $S^1$ geodesic-folded (`[-128, 127] -> [-64, 64]`) PhaseClamp soft-bounded phase steering
/// to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_geodesic_safe(
    query_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_geodesic_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(LMNK_OK)
    })
}

/// Applies PhaseClamp soft-bounded phase steering gated branchlessly by `ACTION_MODE_PHASE_STEER`
/// and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_pointer_gated_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_safe(
            TaggedPtr(tagged_ptr),
            &mut q_aligned,
            &p_aligned,
            max_rad,
        )?;
        Ok(LMNK_OK)
    })
}

/// Applies fused Query + Key PhaseClamp soft-bounded phase steering gated branchlessly by
/// `ACTION_MODE_PHASE_STEER` and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_pointer_gated_fused_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const lmnk_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_fused_safe(
            TaggedPtr(tagged_ptr),
            &mut qk_pair.first,
            &mut qk_pair.second,
            &p_aligned,
            max_rad,
        )?;
        Ok(LMNK_OK)
    })
}

/// Projects an arbitrary raw byte slice (`bytes[0..byte_len]`) onto the 64-head $SO(2)^{64}$ Q7
/// phase manifold (`phases_out[0..64]`) in branchless 8-byte SWAR SplitMix64 chunks via
/// [`project_bytes_swar_to_words_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_gateway_project_bytes_to_q7_64(
    bytes: *const u8,
    byte_len: usize,
    phases_out: *mut lmnk_q7_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        let in_slice: &[u8] = if byte_len == 0 {
            &[]
        } else {
            validate_ptr(bytes)?;
            // SAFETY: Validated non-null for `byte_len > 0`.
            unsafe { slice::from_raw_parts(bytes, byte_len) }
        };
        // SAFETY: `phases_out` was validated non-null above, and `mut_phase64_from_raw` rechecks
        // nullness and constructs an exclusive `&mut [lmnk_q7_t; 64]` (alignment 1).
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        let projected_w = project_bytes_swar_to_words_64(in_slice);
        store_q7_words_64(out_ref, &projected_w);
        Ok(LMNK_OK)
    })
}

/// Projects an `f32` embedding or activation slice (`values[0..num_values]`) onto the 64-head
/// $SO(2)^{64}$ Q7 phase manifold (`phases_out[0..64]`): when `gain > 0.0`, executes the 8-YMM
/// AVX2 SIMD accumulator [`branchless_project_f32_slice_to_q7_64`]; when `gain == 0.0`, executes
/// pairwise $SO(2)$ `atan2` projection [`project_f32_slice_to_q7_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_gateway_project_f32_to_q7_64(
    values: *const f32,
    num_values: usize,
    gain: f32,
    phases_out: *mut lmnk_q7_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        if !branchless_is_finite_f32(gain) || gain < 0.0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let in_slice: &[f32] = if num_values == 0 {
            &[]
        } else {
            validate_ptr(values)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(values, num_values) }
        };
        // SAFETY: `phases_out` was validated non-null above, and `mut_phase64_from_raw` rechecks
        // nullness and constructs an exclusive `&mut [lmnk_q7_t; 64]` (alignment 1).
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        if gain > 0.0 {
            *out_ref = branchless_project_f32_slice_to_q7_64(in_slice, gain);
        } else {
            let projected_w = project_f32_slice_to_words_64(in_slice);
            store_q7_words_64(out_ref, &projected_w);
        }
        Ok(LMNK_OK)
    })
}

/// Synthesizes a 128-byte [`lmnk_gateway_frame_t`] (`GatewayFrame`) and optional 16-byte
/// [`lmnk_causal_report_t`] (`CausalProvenanceReport`) across the C-ABI boundary from
/// the **2-Origin Supertrait Hierarchy**.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_gateway_synthesize_multi_origin_frame(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const lmnk_q7_t,
    fact_confidence: f32,
    fact_mapped_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const lmnk_q7_t,
    persona_strength: f32,
    persona_mapped_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const lmnk_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    frame_out: *mut lmnk_gateway_frame_t,
    report_out: *mut lmnk_causal_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (mut out_frame, out_report_opt) = unsafe {
            (
                FfiHandleMut::from_raw(frame_out)?,
                FfiHandleMut::from_raw_opt(report_out)?,
            )
        };

        // SAFETY: Validated by `synthesize_multi_origin_from_raw_ptrs_with_words`.
        let (frame, _frm_w) = unsafe {
            synthesize_multi_origin_from_raw_ptrs_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_phases,
                fact_confidence,
                fact_mapped_offset48,
                fact_anchor_id,
                persona_phases,
                persona_strength,
                persona_mapped_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        if let Some(mut rep_out) = out_report_opt {
            *rep_out = frame.to_causal_report();
        }
        *out_frame = frame;
        Ok(LMNK_OK)
    })
}

/// Executes closed-loop **2-Origin Composite Manifold Gateway Steering** across the C-ABI boundary
/// in register-resident mode (`0` intermediate `GatewayFrame` copies).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_steer_gateway_frame_safe(
    frame: *const lmnk_gateway_frame_t,
    trail_inout: *mut lmnk_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    report_out: *mut lmnk_causal_report_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(frame)?;
        validate_ptr(trail_inout)?;
        if raw_byte_ranges_overlap(
            frame as usize,
            size_of::<lmnk_gateway_frame_t>(),
            trail_inout as usize,
            size_of::<lmnk_pointer_trail_t>(),
        ) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness and 64B alignment.
        let frame_ref = unsafe { FfiHandleRef::from_raw(frame)? };
        // SAFETY: Validated by `execute_gateway_frame_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_steer_and_optional_kmeans_ffi(
                &frame_ref,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                ptr::null_mut(),
                0,
                0.0,
                ptr::null_mut(),
                report_out,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )?;
        }
        Ok(LMNK_OK)
    })
}

/// Executes a one-shot **Raw-Byte 2-Origin Multi-Blanket Gateway Synthesis + $SO(2)$-Fused
/// PhaseClamp/sub-head router Steering + Optional Online Toroidal $k$-Means / DP-Means Macro Crystallization**
/// across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_bytes: *const u8,
    fact_byte_len: usize,
    fact_confidence: f32,
    fact_mapped_offset48: u64,
    fact_anchor_id: u32,
    persona_bytes: *const u8,
    persona_byte_len: usize,
    persona_strength: f32,
    persona_mapped_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const lmnk_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut lmnk_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut lmnk_phase_anchor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut lmnk_gateway_frame_t,
    report_out: *mut lmnk_causal_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_bytes_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_bytes_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_bytes,
                fact_byte_len,
                fact_confidence,
                fact_mapped_offset48,
                fact_anchor_id,
                persona_bytes,
                persona_byte_len,
                persona_strength,
                persona_mapped_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                frame_out,
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(LMNK_OK)
    })
}

/// Executes a one-shot **Raw-`f32` Embedding 2-Origin Multi-Blanket Gateway Synthesis +
/// $SO(2)$-Fused PhaseClamp/sub-head router Steering + Optional Online Toroidal $k$-Means / DP-Means Macro
/// Crystallization** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `LMNK_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_values: *const f32,
    fact_len: usize,
    fact_projection_gain: f32,
    fact_confidence: f32,
    fact_mapped_offset48: u64,
    fact_anchor_id: u32,
    persona_values: *const f32,
    persona_len: usize,
    persona_projection_gain: f32,
    persona_strength: f32,
    persona_mapped_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const lmnk_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut lmnk_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut lmnk_phase_anchor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut lmnk_gateway_frame_t,
    report_out: *mut lmnk_causal_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> LmnkStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_f32_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_f32_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_values,
                fact_len,
                fact_projection_gain,
                fact_confidence,
                fact_mapped_offset48,
                fact_anchor_id,
                persona_values,
                persona_len,
                persona_projection_gain,
                persona_strength,
                persona_mapped_offset48,
                persona_anchor_id,
                bloom_filter_opt,
                contradiction_threshold,
                bloom_repulsion_threshold,
            )?
        };
        // SAFETY: Validated by `execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi`.
        unsafe {
            execute_gateway_frame_with_words_steer_and_optional_kmeans_ffi(
                &frame,
                &frm_w,
                trail_inout,
                unresolved_damping_u7,
                query_tensor,
                key_tensor,
                head_dim,
                num_heads,
                logits_inout,
                vocab_size,
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                frame_out,
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(LMNK_OK)
    })
}

/// Applies soft phase damping to smooth out transient phase rotation spikes in zero-allocation mode.
///
/// # Safety
/// `query_tensor` must be non-null, 32B aligned, and valid for `head_dim * num_heads` floats.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_apply_soft_damping(
    query_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    damping_factor: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        engine.apply_soft_phase_damping(&mut q_aligned, damping_factor)?;
        Ok(LMNK_OK)
    })
}

/// PhaseStepFilter discrete filter: branchlessly cuts off phase rotation steps falling below `min_step_rad`.
///
/// # Safety
/// `phase_deltas` must be non-null, 16-byte aligned, and valid for `len` elements.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_step_filter(
    phase_deltas: *mut lmnk_q7_t,
    len: usize,
    min_step_rad: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_phase_engine(128, len)?;
        // SAFETY: Validated for non-nullness, 16B alignment, and bounds.
        let mut p_aligned = unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_deltas, len)? };

        engine.phase_step_filter(&mut p_aligned, min_step_rad)?;
        Ok(LMNK_OK)
    })
}

/// Applies Lazy Phase Unwrapping (Query-Only Relative Phase Shift) override on a Query tensor.
///
/// # Safety
/// `query_tensor` and `context_accum` must be non-null, 32B aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_lazy_unwrap_override(
    query_tensor: *mut f32,
    context_accum: *const f32,
    head_dim: usize,
    num_heads: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(context_accum, 32)?;
        let (engine, total_q_len) = init_phase_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut pair = unsafe {
            DisjointRefMut::<f32, 32, 32>::from_raw_parts(
                context_accum,
                num_heads,
                query_tensor,
                total_q_len,
            )?
        };
        let (c_aligned, q_aligned) = pair.as_ref_mut();

        engine.lazy_unwrap_override(q_aligned, c_aligned)?;
        Ok(LMNK_OK)
    })
}

/// Executes norm-guarded Slerp phase stabilization on a state vector in zero-allocation mode.
///
/// # Safety
/// `tensor` must be non-null, 32B aligned, and valid for `dim` floats.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_slerp_norm_guard_stable(
    tensor: *mut f32,
    dim: usize,
    norm_lower: f32,
    norm_upper: f32,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(tensor, 32)?;
        let engine = PhaseSteeringEngine::new(128, 1)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut t_aligned = unsafe { AlignedSliceMut32::from_raw_parts_mut(tensor, dim)? };

        engine.slerp_norm_guard_stable(&mut t_aligned, norm_lower, norm_upper)?;
        Ok(LMNK_OK)
    })
}

/// Applies a fused inline logit phase shift prior to the LM Head Softmax layer.
///
/// # Safety
/// `logits` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_fused_logit_shift(
    logits: *mut f32,
    vocab_size: usize,
    phase_deltas: *const lmnk_q7_t,
    num_heads: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(logits, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_phase_engine(128, num_heads)?;
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let (mut l_aligned, p_aligned) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut(logits, vocab_size)?,
                AlignedSlice16::from_raw_parts(phase_deltas, num_heads)?,
            )
        };

        engine.fused_logit_shift(&mut l_aligned, &p_aligned)?;
        Ok(LMNK_OK)
    })
}

/// Computes Orthogonal Procrustes phase delta alignment in $SO(2)^{d_{\mathrm{head}}/2}$ subspace
/// between `source` and `target` state representations.
///
/// # Safety
/// `source` (32B), `target` (32B), and `phase_out` (16B) must be non-null and aligned.
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_compute_procrustes_delta(
    source: *const f32,
    target: *const f32,
    dim: usize,
    phase_out: *mut lmnk_q7_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(source, 32)?;
        validate_ptr_aligned(target, 32)?;
        validate_ptr(phase_out)?;
        let (head_dim, num_heads) = resolve_flat_tensor_heads(dim)?;
        let engine = PhaseSteeringEngine::new(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let (src_aligned, tgt_aligned) = unsafe {
            (
                AlignedSlice32::from_raw_parts(source, dim)?,
                AlignedSlice32::from_raw_parts(target, dim)?,
            )
        };
        // SAFETY: Validated for non-nullness, 16B alignment, and bounds.
        let mut out_aligned =
            unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_out, num_heads as usize)? };

        engine.compute_procrustes_phase_delta(&src_aligned, &tgt_aligned, &mut out_aligned)?;
        Ok(LMNK_OK)
    })
}

/// Destroys a PhaseSteering kernel instance and releases its heap allocation.
///
/// # Safety
/// `kernel` must be `NULL` or a handle returned by [`lmnk_phase_steering_create`].
#[no_mangle]
pub unsafe extern "C" fn lmnk_phase_steering_destroy(
    kernel: LmnkPhaseSteeringHandle,
) -> LmnkStatus {
    if kernel.is_null() {
        return LMNK_OK;
    }
    if !is_aligned(kernel) {
        return LMNK_ERR_ALIGNMENT;
    }
    ffi_boundary!({
        // SAFETY: `kernel` was allocated via `Box::into_raw` in `lmnk_phase_steering_create`.
        unsafe {
            let _ = Box::from_raw(kernel.cast::<PhaseSteeringEngine>());
        }
        Ok(LMNK_OK)
    })
}

/// Dispatches sub-head group phase gains (Fact, Logic, Explore) in a single forward pass.
///
/// # Safety
/// `query_tensor` (32B aligned) and `subhead_cfg` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn lmnk_subhead_dispatch_single_pass(
    query_tensor: *mut f32,
    subhead_cfg: *const lmnk_subhead_config_t,
    head_dim: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        // SAFETY: Validated for non-nullness and 16B alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(subhead_cfg)? };
        if head_dim == 0 || (head_dim & 1) != 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let total_heads = (cfg.num_fact_heads as usize)
            .checked_add(cfg.num_logic_heads as usize)
            .and_then(|v| v.checked_add(cfg.num_explore_heads as usize))
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        if total_heads == 0 {
            return Ok(LMNK_OK);
        }

        let expected_len = head_dim
            .checked_mul(total_heads)
            .ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, expected_len)? };

        dispatch_single_pass(&mut q_aligned, &cfg, head_dim)?;
        Ok(LMNK_OK)
    })
}

/// Evaluates Rank-1 residual Dynamic Rank-1 outer product: $\mathbf{y} = \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_out` (32B), `x_in` (32B), and `rank1_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn lmnk_rank1_eval(
    y_out: *mut f32,
    x_in: *const f32,
    rank1_desc: *const lmnk_rank1_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_out, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(rank1_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard =
            unsafe { DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_out, dim_out)? };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        eval_rank1(y_slice, x_slice, &desc)?;
        Ok(LMNK_OK)
    })
}

/// Evaluates Rank-1 residual Dynamic Rank-1 outer product and accumulates directly into `y_inout` in-place:
/// $\mathbf{y} \gets \mathbf{y} + \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_inout` (32B), `x_in` (32B), and `rank1_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn lmnk_rank1_eval_fused(
    y_inout: *mut f32,
    x_in: *const f32,
    rank1_desc: *const lmnk_rank1_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_inout, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(rank1_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard = unsafe {
            DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_inout, dim_out)?
        };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        eval_rank1_fused(y_slice, x_slice, &desc)?;
        Ok(LMNK_OK)
    })
}

/// L2P-SVD: Training-free projection of fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$,
/// $W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$) into an Rank-1 residual Rank-1 descriptor and Q7 phase profile (Section 5.2).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn lmnk_lora_project(
    lora_a: *const f32,
    lora_b: *const f32,
    rank: usize,
    dim_in: usize,
    dim_out: usize,
    rank1_out: *mut lmnk_rank1_descriptor_t,
    phase_profile_out: *mut lmnk_q7_t,
) -> LmnkStatus {
    ffi_boundary!({
        validate_ptr(lora_a)?;
        validate_ptr(lora_b)?;
        validate_ptr(phase_profile_out)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut rank1_desc = unsafe { FfiHandleMut::from_raw(rank1_out)? };
        if rank == 0 || dim_in == 0 || dim_out == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }

        let a_len = rank.checked_mul(dim_in).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        let b_len = dim_out.checked_mul(rank).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;

        if is_ptr_aligned_to(lora_a, 32)
            && is_ptr_aligned_to(lora_b, 32)
            && is_ptr_aligned_to(phase_profile_out, 16)
            && ((dim_out & 1) == 0)
        {
            let num_heads = (dim_out / 128).clamp(1, MAX_ATTENTION_HEADS);
            // SAFETY: `lora_a` and `lora_b` are non-null, verified 32-byte aligned, and
            // `a_len`/`b_len` were computed with checked multiplication above.
            let (a_aligned, b_aligned) = unsafe {
                (
                    AlignedSlice32::from_raw_parts(lora_a, a_len)?,
                    AlignedSlice32::from_raw_parts(lora_b, b_len)?,
                )
            };
            // SAFETY: `phase_profile_out` is non-null, verified 16-byte aligned above, and
            // `num_heads` is clamped to `1..=MAX_ATTENTION_HEADS`.
            let mut p_aligned =
                unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_profile_out, num_heads)? };

            project_lora(
                &a_aligned,
                &b_aligned,
                rank,
                dim_in,
                dim_out,
                &mut rank1_desc,
                &mut p_aligned,
            )?;
            return Ok(LMNK_OK);
        }

        // SAFETY: Validated non-null and 4B aligned above.
        let (a_slice, b_slice) = unsafe {
            (
                slice::from_raw_parts(lora_a, a_len),
                slice::from_raw_parts(lora_b, b_len),
            )
        };

        let norm_u_sq = if rank == 1 {
            sum_sq_f32_simd8(&b_slice[..dim_out])
        } else {
            let mut total_u = 0.0_f32;
            let mut i = 0usize;
            while i < dim_out {
                let val = b_slice[i * rank];
                total_u += val * val;
                i += 1;
            }
            total_u
        };
        let norm_v_sq = sum_sq_f32_simd8(&a_slice[..dim_in]);

        let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
        rank1_desc.layer_idx = 0;
        rank1_desc.rank = 1;
        rank1_desc.u_vector_ptr = lora_b as u64;
        rank1_desc.v_vector_ptr = lora_a as u64;
        rank1_desc.gain_scalar_ptr = 0;
        rank1_desc.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

        let phase_val = if dim_out >= 2 {
            branchless_atan2_to_q7_fast(b_slice[rank], b_slice[0])
        } else {
            0
        };
        // SAFETY: `phase_profile_out` is validated non-null above.
        unsafe { *phase_profile_out = phase_val };
        Ok(LMNK_OK)
    })
}
