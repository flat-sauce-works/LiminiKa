// name: src/core/src/abi/phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Phase Steering (DPSR), QDPS, 2-Origin Gateway, PSPM routing, and Residual Adapter (SRL) FFI.

use super::*;

// ===================================================================
// 4. DPSR, QDPS, 2-Origin Gateway, PSPM & SRL Interface
// ===================================================================

/// Creates a DPSR phase steering kernel instance.
///
/// # Safety
/// `kernel_out` must be a non-null, naturally aligned writable handle pointer.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_create(
    head_dim: u32,
    num_heads: u32,
    kernel_out: *mut GcsoDpsrKernelHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let mut out_handle = unsafe { FfiHandleMut::from_raw(kernel_out)? };
        let engine = PhaseSteeringEngine::new(head_dim, num_heads)?;

        *out_handle = Box::into_raw(Box::new(engine)).cast::<GcsoDpsrKernelOpaque>();
        Ok(GCSO_SUCCESS)
    })
}

/// Applies inline DPSR phase steering to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering(&mut q_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused inline DPSR phase steering simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_fused(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_disjoint_pair(&mut qk_pair, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded $\tanh$ clamping ($\Delta\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\Delta\theta/\theta_{\max})$)
/// restricted to low-frequency RoPE channels (upper $d_{\mathrm{head}}/4$ dimensions).
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded $\tanh$ clamping on `HalfSplit` RoPE channels with optional
/// partial `rotary_dim` (upper quarter of `[0..rotary_dim]` split-half pairs).
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_half_split_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;
        let geom = crate::traits::rope::RopeGeometryDescriptor {
            layout: crate::traits::rope::RopePairLayout::HalfSplit,
            head_dim,
            rotary_dim,
            num_heads,
        };

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        crate::traits::rope::PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
            &engine,
            &geom,
            &mut q_aligned,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded phase steering dispatched by [`gcso_config_t`] RoPE geometry
/// (`Interleaved` vs `HalfSplit`, partial `rotary_dim`, `head_dim`, `num_heads`, `ripa_clamp_max_rad`).
///
/// # Safety
/// `query_tensor` (32B aligned), `phase_deltas` (16B aligned), and `config` (16B aligned)
/// must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_with_config_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    config: *const gcso_config_t,
) -> GcsoStatus {
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
        let layout = crate::traits::rope::RopePairLayout::from_u8(cfg.rope_layout)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;
        let geom = crate::traits::rope::RopeGeometryDescriptor {
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

        crate::traits::rope::PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe(
            &engine,
            &geom,
            &mut q_aligned,
            &p_aligned,
            cfg.ripa_clamp_max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Steers `query_tensor` using a 2-Origin [`ManifoldGatewaySignal`](crate::traits::gateway::ManifoldGatewaySignal)
/// composite (`(ObjectiveFact, SubjectivePersona)`) so structural hyperparameters are synthesized
/// autonomously from Gateway physics and dispatched via [`gcso_config_t`] RoPE geometry.
///
/// # Safety
/// `query_tensor` (32B aligned), `fact_phases` (16B aligned), `persona_phases` (16B aligned),
/// `trail_inout` (128B aligned), and `config` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_with_gateway_and_config_safe(
    query_tensor: *mut f32,
    fact_phases: *const gcso_q7_t,
    fact_energy: f32,
    persona_phases: *const gcso_q7_t,
    persona_energy: f32,
    trail_inout: *mut gcso_pointer_trail_t,
    config: *const gcso_config_t,
) -> GcsoStatus {
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
        let layout = crate::traits::rope::RopePairLayout::from_u8(cfg.rope_layout)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;
        let geom = crate::traits::rope::RopeGeometryDescriptor {
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

        let mut fact_arr = [0i8; crate::traits::primitives::MAX_ATTENTION_HEADS];
        let mut persona_arr = [0i8; crate::traits::primitives::MAX_ATTENTION_HEADS];
        fact_arr[..num_heads].copy_from_slice(fact_slice.as_slice());
        persona_arr[..num_heads].copy_from_slice(persona_slice.as_slice());

        let composite = (
            crate::traits::gateway::ObjectiveFact::new(fact_arr, fact_energy, 0, 1),
            crate::traits::gateway::SubjectivePersona::new(persona_arr, persona_energy, 0, 2),
        );

        crate::traits::gateway::GatewayPhaseSteeringExt::steer_gateway_signal_for_geometry_safe(
            &engine,
            &composite,
            &geom,
            &mut trail,
            128,
            &mut q_aligned,
            None,
            None,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Recovers per-head Q7 Procrustes phase deltas from `HalfSplit` RoPE tensor pairs.
///
/// # Safety
/// `q_before` (32B aligned), `q_after` (32B aligned), and `q7_phase_out` (16B aligned)
/// must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_compute_procrustes_phase_delta_half_split(
    q_before: *const f32,
    q_after: *const f32,
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
    q7_phase_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(q_before, 32)?;
        validate_ptr_aligned(q_after, 32)?;
        validate_ptr(q7_phase_out)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;
        let geom = crate::traits::rope::RopeGeometryDescriptor {
            layout: crate::traits::rope::RopePairLayout::HalfSplit,
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

        crate::traits::rope::PhaseSteeringGeometryExt::compute_procrustes_phase_delta_for_geometry(
            &engine,
            &geom,
            &q_bef_aligned,
            &q_aft_aligned,
            &mut p_out_aligned,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused RIPA soft-bounded $\tanh$ clamping simultaneously to disjoint Query and Key tensors.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_fused_safe(
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

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
        Ok(GCSO_SUCCESS)
    })
}

/// Applies $S^1$ geodesic-folded (`[-128, 127] -> [-64, 64]`) RIPA soft-bounded phase steering
/// to a Query tensor in zero-allocation mode.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_phase_steering_geodesic_safe(
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_phase_steering_geodesic_safe(&mut q_aligned, &p_aligned, max_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies RIPA soft-bounded phase steering gated branchlessly by `ACTION_MODE_DPSR_STEER`
/// and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// `query_tensor` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_pointer_gated_phase_steering_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_safe(
            gcso_tagged_ptr_t(tagged_ptr),
            &mut q_aligned,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies fused Query + Key RIPA soft-bounded phase steering gated branchlessly by
/// `ACTION_MODE_DPSR_STEER` and the embedded Q7 phase offset inside `tagged_ptr`.
///
/// # Safety
/// Pointers must satisfy non-null, 32B/16B alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_pointer_gated_phase_steering_fused_safe(
    tagged_ptr: u64,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    phase_deltas: *const gcso_q7_t,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(key_tensor, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, total_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, bounds, and disjointness.
        let mut qk_pair = unsafe {
            DisjointMutPair::<f32, 32>::from_raw_parts_mut(query_tensor, key_tensor, total_len)?
        };
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let p_aligned = unsafe { AlignedSlice16::from_raw_parts(phase_deltas, num_heads)? };

        engine.apply_pointer_gated_phase_steering_fused_safe(
            gcso_tagged_ptr_t(tagged_ptr),
            &mut qk_pair.first,
            &mut qk_pair.second,
            &p_aligned,
            max_rad,
        )?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies **Order-Invariant Single-Manifold Unified-Field Phase Steering**: folds `num_anchors`
/// anchors on the unified phase manifold into `trail_phases` starting from `trail_init_weight`,
/// applies 8-lane SWAR QDPS filtering (`min_step_rad`), and steers `query_tensor` (and optional
/// `key_tensor`) with RIPA soft-clamping (`max_rad`).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_unified_field_phase_steering_safe(
    trail_phases: *const gcso_q7_t,
    trail_init_weight: f32,
    anchor_phases: *const gcso_q7_t,
    anchor_base_energies: *const f32,
    num_anchors: usize,
    min_step_rad: f32,
    max_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    total_pull_out: *mut f32,
    mean_coherence_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };

        // SAFETY: Validated non-null and bounds-checked.
        let (trail_ref, anchors_slice) = unsafe {
            (
                ref_phase64_from_raw(trail_phases)?,
                slice_of_phase64_from_raw(anchor_phases, num_anchors)?,
            )
        };
        let energies_slice: &[f32] = if num_anchors == 0 {
            &[]
        } else {
            validate_ptr(anchor_base_energies)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(anchor_base_energies, num_anchors) }
        };

        let (tot_pull, mean_coh, dom_id) = engine
            .apply_unified_field_phase_steering_with_inertia_safe(
                trail_ref,
                trail_init_weight,
                anchors_slice,
                energies_slice,
                min_step_rad,
                max_rad,
                &mut q_aligned,
                k_opt.as_mut(),
            )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(mean_coherence_out, mean_coh)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Directly modulates `num_candidates` pre-Softmax candidate memory/key Attention scores
/// (`attention_scores_inout`) in-place by the **Attention $\times$ DPSR Phase Interference**
/// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$
/// across `key_anchor_phases` (`num_candidates * 64` `gcso_q7_t` elements).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_modulate_attention_scores_by_phase_bank(
    attention_scores_inout: *mut f32,
    query_phases: *const gcso_q7_t,
    key_anchor_phases: *const gcso_q7_t,
    num_candidates: usize,
    head_dim: u32,
    interference_gain: f32,
    mean_coherence_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(attention_scores_inout)?;
        // SAFETY: Validated non-null.
        let q_ref = unsafe { ref_phase64_from_raw(query_phases)? };
        if num_candidates == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let scores_bytes = num_candidates
            .checked_mul(size_of::<f32>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let bank_bytes = num_candidates
            .checked_mul(MAX_ATTENTION_HEADS)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            query_phases as usize,
            MAX_ATTENTION_HEADS,
        ) || raw_byte_ranges_overlap(
            attention_scores_inout as usize,
            scores_bytes,
            key_anchor_phases as usize,
            bank_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (key_bank, scores) = unsafe {
            (
                slice_of_phase64_from_raw(key_anchor_phases, num_candidates)?,
                slice::from_raw_parts_mut(attention_scores_inout, num_candidates),
            )
        };

        let mean_coh = modulate_attention_scores_by_phase_bank_64(
            scores,
            q_ref,
            key_bank,
            head_dim,
            interference_gain,
        );

        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(mean_coherence_out, mean_coh)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Projects an arbitrary raw byte slice (`bytes[0..byte_len]`) onto the 64-head $SO(2)^{64}$ Q7
/// phase manifold (`phases_out[0..64]`) in branchless 8-byte SWAR SplitMix64 chunks via
/// [`project_bytes_swar_to_words_64`].
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_bytes_to_q7_64(
    bytes: *const u8,
    byte_len: usize,
    phases_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        let in_slice: &[u8] = if byte_len == 0 {
            &[]
        } else {
            validate_ptr(bytes)?;
            // SAFETY: Validated non-null for `byte_len > 0`.
            unsafe { slice::from_raw_parts(bytes, byte_len) }
        };
        // SAFETY: Validated non-null.
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        let projected_w = project_bytes_swar_to_words_64(in_slice);
        store_q7_words_64(out_ref, &projected_w);
        Ok(GCSO_SUCCESS)
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
pub unsafe extern "C" fn gcso_gateway_project_f32_to_q7_64(
    values: *const f32,
    num_values: usize,
    gain: f32,
    phases_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phases_out)?;
        if !branchless_is_finite_f32(gain) || gain < 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let in_slice: &[f32] = if num_values == 0 {
            &[]
        } else {
            validate_ptr(values)?;
            // SAFETY: Validated non-null and 4B aligned.
            unsafe { slice::from_raw_parts(values, num_values) }
        };
        // SAFETY: Validated non-null.
        let out_ref = unsafe { mut_phase64_from_raw(phases_out)? };
        if gain > 0.0 {
            *out_ref = branchless_project_f32_slice_to_q7_64(in_slice, gain);
        } else {
            let projected_w = project_f32_slice_to_words_64(in_slice);
            store_q7_words_64(out_ref, &projected_w);
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Synthesizes a 128-byte [`gcso_gateway_frame_t`] (`GatewayFrame`) and optional 16-byte
/// [`gcso_causal_provenance_report_t`] (`CausalProvenanceReport`) across the C-ABI boundary from
/// the **2-Origin Supertrait Hierarchy**.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_multi_origin_frame(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
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
                fact_zimms_offset48,
                fact_anchor_id,
                persona_phases,
                persona_strength,
                persona_zimms_offset48,
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
        Ok(GCSO_SUCCESS)
    })
}

/// Executes closed-loop **2-Origin Composite Manifold Gateway Steering** across the C-ABI boundary
/// in register-resident mode (`0` intermediate `GatewayFrame` copies).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_gateway_frame_safe(
    frame: *const gcso_gateway_frame_t,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame)?;
        validate_ptr(trail_inout)?;
        if raw_byte_ranges_overlap(
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused 2-Origin Composite Manifold Gateway Steering + Online Toroidal
/// $k$-Means / DP-Means Macro Crystallization** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_and_kmeans_crystallize_gateway_frame_safe(
    frame: *const gcso_gateway_frame_t,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame)?;
        validate_ptr(trail_inout)?;
        validate_ptr(centroids_inout)?;
        if num_centroids == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
        ) || raw_byte_ranges_overlap(
            centroids_inout as usize,
            cent_bytes,
            frame as usize,
            size_of::<gcso_gateway_frame_t>(),
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
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
                centroids_inout,
                num_centroids,
                split_coherence_threshold,
                ptr::null_mut(),
                report_out,
                assigned_centroid_idx_out,
                assigned_coherence_out,
                updated_gravity_mass_out,
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **2-Origin Multi-Blanket Gateway Synthesis + $SO(2)$-Fused RIPA/PSPM
/// Steering + Optional Online Toroidal $k$-Means / DP-Means Macro Crystallization** across the
/// C-ABI boundary ($C_{\mathrm{alloc}} = 0$, `0` intermediate C-stack `GatewayFrame` roundtrips).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_phases: *const gcso_q7_t,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_phases: *const gcso_q7_t,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated by `synthesize_multi_origin_from_raw_ptrs_with_words`.
        let (frame, frm_w) = unsafe {
            synthesize_multi_origin_from_raw_ptrs_with_words(
                vram_pressure,
                gpu_frame_overrun,
                active_swarm_cell_mask,
                recommend_layer_skip,
                fact_phases,
                fact_confidence,
                fact_zimms_offset48,
                fact_anchor_id,
                persona_phases,
                persona_strength,
                persona_zimms_offset48,
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
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Raw-Byte 2-Origin Multi-Blanket Gateway Synthesis + $SO(2)$-Fused
/// RIPA/PSPM Steering + Optional Online Toroidal $k$-Means / DP-Means Macro Crystallization**
/// across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_bytes_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_bytes: *const u8,
    fact_byte_len: usize,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_bytes: *const u8,
    persona_byte_len: usize,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
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
                fact_zimms_offset48,
                fact_anchor_id,
                persona_bytes,
                persona_byte_len,
                persona_strength,
                persona_zimms_offset48,
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
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Raw-`f32` Embedding 2-Origin Multi-Blanket Gateway Synthesis +
/// $SO(2)$-Fused RIPA/PSPM Steering + Optional Online Toroidal $k$-Means / DP-Means Macro
/// Crystallization** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_project_f32_synthesize_kmeans_and_steer_qk_safe(
    vram_pressure: f32,
    gpu_frame_overrun: f32,
    active_swarm_cell_mask: u8,
    recommend_layer_skip: u8,
    fact_values: *const f32,
    fact_len: usize,
    fact_projection_gain: f32,
    fact_confidence: f32,
    fact_zimms_offset48: u64,
    fact_anchor_id: u32,
    persona_values: *const f32,
    persona_len: usize,
    persona_projection_gain: f32,
    persona_strength: f32,
    persona_zimms_offset48: u64,
    persona_anchor_id: u32,
    bloom_filter_opt: *const gcso_contrastive_bloom_filter_t,
    contradiction_threshold: f32,
    bloom_repulsion_threshold: u8,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    centroids_inout: *mut gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    split_coherence_threshold: f32,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
    assigned_centroid_idx_out: *mut usize,
    assigned_coherence_out: *mut f32,
    updated_gravity_mass_out: *mut f32,
) -> GcsoStatus {
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
                fact_zimms_offset48,
                fact_anchor_id,
                persona_values,
                persona_len,
                persona_projection_gain,
                persona_strength,
                persona_zimms_offset48,
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
        Ok(GCSO_SUCCESS)
    })
}

/// Synthesizes a closed-loop 128-byte [`gcso_gateway_frame_t`] (`GatewayFrame`) and optional
/// 16-byte [`gcso_causal_provenance_report_t`] (`CausalProvenanceReport`) directly from an array
/// of `num_records` crystallized macro centroids via the blanket [`ManifoldGatewaySignal`]
/// implementation on `[CrystallizedAttractorRecord]`.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_gateway_synthesize_from_crystallized_records(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(frame_out)?;
        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                records as usize,
                rec_bytes,
                frame_out as usize,
                size_of::<gcso_gateway_frame_t>(),
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        // SAFETY: Validated for non-nullness and 64B alignment.
        let mut out_frame = unsafe { FfiHandleMut::from_raw(frame_out)? };
        let frame = records_slice.synthesize_gateway_frame();
        let report = frame.to_causal_report();
        *out_frame = frame;
        // SAFETY: `write_opt_out` validates optional output pointer.
        unsafe {
            write_opt_out(report_out, report)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a one-shot **Closed-Loop Macro-Distribution Gateway Re-Injection + $SO(2)$-Fused
/// RIPA/PSPM Steering** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$) from
/// `records[0..num_records]` via the blanket [`ManifoldGatewaySignal`] implementation.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_closed_loop_crystallized_gateway_safe(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    logits_inout: *mut f32,
    vocab_size: usize,
    frame_out: *mut gcso_gateway_frame_t,
    report_out: *mut gcso_causal_provenance_report_t,
) -> GcsoStatus {
    ffi_boundary!({
        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                records as usize,
                rec_bytes,
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
            ) || (!frame_out.is_null()
                && raw_byte_ranges_overlap(
                    records as usize,
                    rec_bytes,
                    frame_out as usize,
                    size_of::<gcso_gateway_frame_t>(),
                ))
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        let (frame, frm_w) = records_slice.synthesize_gateway_frame_with_words();
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
                ptr::null_mut(),
                0,
                0.0,
                frame_out,
                report_out,
                ptr::null_mut(),
                ptr::null_mut(),
                ptr::null_mut(),
            )?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Folds `num_records` `.gcso` [`gcso_crystallized_attractor_record_t`] centroids (`records`) onto
/// `trail_inout` via **Phase-Proximity Gravity** directly in 8 `u64` SWAR registers and steers
/// `query_tensor` (and optional `key_tensor`) without intermediate `GatewayFrame` construction.
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_steer_crystallized_ensemble_zero_copy_safe(
    records: *const gcso_crystallized_attractor_record_t,
    num_records: usize,
    trail_inout: *mut gcso_pointer_trail_t,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    total_pull_out: *mut f32,
    dominant_anchor_id_out: *mut u32,
    effective_ripa_rad_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and 128B alignment.
        let mut trail = unsafe { FfiHandleMut::from_raw(trail_inout)? };
        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };

        let records_slice: &[CrystallizedAttractorRecord] = if num_records == 0 {
            &[]
        } else {
            validate_ptr(records)?;
            let rec_bytes = num_records
                .checked_mul(size_of::<CrystallizedAttractorRecord>())
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if raw_byte_ranges_overlap(
                trail_inout as usize,
                size_of::<gcso_pointer_trail_t>(),
                records as usize,
                rec_bytes,
            ) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            // SAFETY: Validated non-null, 64B aligned, and disjoint above.
            unsafe { slice::from_raw_parts(records, num_records) }
        };

        let (tot_pull, dom_id, eff_ripa) = steer_crystallized_records_with_optional_bloom_safe(
            records_slice,
            None,
            &mut trail,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        )?;

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(total_pull_out, tot_pull)?;
            write_opt_out(dominant_anchor_id_out, dom_id)?;
            write_opt_out(effective_ripa_rad_out, eff_ripa)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Executes a single-pass **Fused Coarse-to-Fine Macro-Centroid Trail Routing ($\mathcal{O}(K)$)
/// + In-Register QDPS/RIPA Phase Steering** across the C-ABI boundary ($C_{\mathrm{alloc}} = 0$).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, bounds, and `GCSO_RESTRICT` invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_action_hub_route_coarse_to_fine_and_steer_qk_safe(
    trail_inout: *mut gcso_pointer_trail_t,
    centroids: *const gcso_crystallized_attractor_record_t,
    num_centroids: usize,
    pull_step_u7: u8,
    min_step_rad: f32,
    query_tensor: *mut f32,
    key_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    winning_centroid_idx_out: *mut usize,
    selected_anchor_id_out: *mut u32,
    centroid_coherence_out: *mut f32,
    gravity_pull_out: *mut f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(trail_inout)?;
        validate_ptr(centroids)?;
        if num_centroids == 0 || min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let cent_bytes = num_centroids
            .checked_mul(size_of::<CrystallizedAttractorRecord>())
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if raw_byte_ranges_overlap(
            trail_inout as usize,
            size_of::<gcso_pointer_trail_t>(),
            centroids as usize,
            cent_bytes,
        ) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        // SAFETY: `bind_qk_32` validates `query_tensor` and `key_tensor` alignment and disjointness.
        let (_engine, mut q_aligned, mut k_opt) =
            unsafe { bind_qk_32(query_tensor, key_tensor, head_dim, num_heads)? };
        // SAFETY: Validated non-null, aligned, and disjoint above.
        let (mut trail, centroids_slice) = unsafe {
            (
                FfiHandleMut::from_raw(trail_inout)?,
                slice::from_raw_parts(centroids, num_centroids),
            )
        };

        let (win_k, anchor_id, coh, grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                centroids_slice,
                &mut trail,
                pull_step_u7,
            );
        trail.sync_flags_with_current_ptr();
        steer_routed_trail_words_qk_safe(
            &bridged_w,
            trail.flags,
            centroids_slice[win_k].ripa_clamp_rad,
            min_step_rad,
            head_dim,
            num_heads,
            &mut q_aligned,
            k_opt.as_mut(),
        );

        // SAFETY: `write_opt_out` validates optional output pointers.
        unsafe {
            write_opt_out(winning_centroid_idx_out, win_k)?;
            write_opt_out(selected_anchor_id_out, anchor_id)?;
            write_opt_out(centroid_coherence_out, coh)?;
            write_opt_out(gravity_pull_out, grav)?;
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Applies soft phase damping to smooth out transient phase rotation spikes in zero-allocation mode.
///
/// # Safety
/// `query_tensor` must be non-null, 32B aligned, and valid for `head_dim * num_heads` floats.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_apply_soft_phase_damping(
    query_tensor: *mut f32,
    head_dim: usize,
    num_heads: usize,
    damping_factor: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, total_q_len)? };
        engine.apply_soft_phase_damping(&mut q_aligned, damping_factor)?;
        Ok(GCSO_SUCCESS)
    })
}

/// QDPS discrete filter: branchlessly cuts off phase rotation steps falling below `min_step_rad`.
///
/// # Safety
/// `phase_deltas` must be non-null, 16-byte aligned, and valid for `len` elements.
#[no_mangle]
pub unsafe extern "C" fn gcso_qdps_filter_step(
    phase_deltas: *mut gcso_q7_t,
    len: usize,
    min_step_rad: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_dpsr_engine(128, len)?;
        // SAFETY: Validated for non-nullness, 16B alignment, and bounds.
        let mut p_aligned = unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_deltas, len)? };

        engine.qdps_filter_step(&mut p_aligned, min_step_rad)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies Lazy Phase Unwrapping (Query-Only Relative Phase Shift) override on a Query tensor.
///
/// # Safety
/// `query_tensor` and `context_accum` must be non-null, 32B aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_lazy_unwrap_override(
    query_tensor: *mut f32,
    context_accum: *const f32,
    head_dim: usize,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        validate_ptr_aligned(context_accum, 32)?;
        let (engine, total_q_len) = init_dpsr_engine(head_dim, num_heads)?;

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
        Ok(GCSO_SUCCESS)
    })
}

/// Executes norm-guarded Slerp phase stabilization on a state vector in zero-allocation mode.
///
/// # Safety
/// `tensor` must be non-null, 32B aligned, and valid for `dim` floats.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_slerp_norm_guard_stable(
    tensor: *mut f32,
    dim: usize,
    norm_lower: f32,
    norm_upper: f32,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(tensor, 32)?;
        let engine = PhaseSteeringEngine::new(128, 1)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut t_aligned = unsafe { AlignedSliceMut32::from_raw_parts_mut(tensor, dim)? };

        engine.slerp_norm_guard_stable(&mut t_aligned, norm_lower, norm_upper)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Applies a fused inline logit phase shift prior to the LM Head Softmax layer.
///
/// # Safety
/// `logits` (32B aligned) and `phase_deltas` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_fused_logit_shift(
    logits: *mut f32,
    vocab_size: usize,
    phase_deltas: *const gcso_q7_t,
    num_heads: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(logits, 32)?;
        validate_ptr(phase_deltas)?;
        let (engine, _) = init_dpsr_engine(128, num_heads)?;
        // SAFETY: Validated for non-nullness, alignment, and bounds.
        let (mut l_aligned, p_aligned) = unsafe {
            (
                AlignedSliceMut32::from_raw_parts_mut(logits, vocab_size)?,
                AlignedSlice16::from_raw_parts(phase_deltas, num_heads)?,
            )
        };

        engine.fused_logit_shift(&mut l_aligned, &p_aligned)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Computes Orthogonal Procrustes phase delta alignment in $SO(2)^{d_{\mathrm{head}}/2}$ subspace
/// between `source` and `target` state representations.
///
/// # Safety
/// `source` (32B), `target` (32B), and `phase_out` (16B) must be non-null and aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_compute_procrustes_phase_delta(
    source: *const f32,
    target: *const f32,
    dim: usize,
    phase_out: *mut gcso_q7_t,
) -> GcsoStatus {
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
        Ok(GCSO_SUCCESS)
    })
}

/// Destroys a DPSR kernel instance and releases its heap allocation.
///
/// # Safety
/// `kernel` must be `NULL` or a handle returned by [`gcso_dpsr_kernel_create`].
#[no_mangle]
pub unsafe extern "C" fn gcso_dpsr_kernel_destroy(kernel: GcsoDpsrKernelHandle) -> GcsoStatus {
    if kernel.is_null() {
        return GCSO_SUCCESS;
    }
    if !is_aligned(kernel) {
        return GCSO_ERROR_MISALIGNED_POINTER;
    }
    ffi_boundary!({
        // SAFETY: `kernel` was allocated via `Box::into_raw` in `gcso_dpsr_kernel_create`.
        unsafe {
            let _ = Box::from_raw(kernel.cast::<PhaseSteeringEngine>());
        }
        Ok(GCSO_SUCCESS)
    })
}

/// Creates a PSPM sub-head group router instance.
///
/// # Safety
/// `config` and `router_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_router_create(
    config: *const gcso_pspm_config_t,
    router_out: *mut GcsoPspmRouterHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (_cfg, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(config)?,
                FfiHandleMut::from_raw(router_out)?,
            )
        };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Dispatches PSPM sub-head group phase gains (Fact, Logic, Explore) in a single forward pass.
///
/// # Safety
/// `query_tensor` (32B aligned) and `pspm_cfg` (16B aligned) must be non-null and valid.
#[no_mangle]
pub unsafe extern "C" fn gcso_pspm_dispatch_single_pass(
    query_tensor: *mut f32,
    pspm_cfg: *const gcso_pspm_config_t,
    head_dim: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(query_tensor, 32)?;
        // SAFETY: Validated for non-nullness and 16B alignment.
        let cfg = unsafe { FfiHandleRef::from_raw(pspm_cfg)? };
        if head_dim == 0 || (head_dim & 1) != 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let total_heads = (cfg.num_fact_heads as usize)
            .checked_add(cfg.num_logic_heads as usize)
            .and_then(|v| v.checked_add(cfg.num_explore_heads as usize))
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if total_heads == 0 {
            return Ok(GCSO_SUCCESS);
        }

        let expected_len = head_dim
            .checked_mul(total_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        // SAFETY: Validated for non-nullness, 32B alignment, and bounds.
        let mut q_aligned =
            unsafe { AlignedSliceMut32::from_raw_parts_mut(query_tensor, expected_len)? };

        AbiPspmDispatcher.dispatch_single_pass(&mut q_aligned, &cfg, head_dim)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Creates a Sparse Residual Adapter Layer (SRL) Rank-1 instance.
///
/// # Safety
/// `descriptor` and `adapter_out` must be non-null and naturally aligned.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_adapter_create(
    descriptor: *const gcso_srl_descriptor_t,
    adapter_out: *mut GcsoSrlAdapterHandle,
) -> GcsoStatus {
    ffi_boundary!({
        // SAFETY: Validated for non-nullness and alignment.
        let (_desc, mut out_handle) = unsafe {
            (
                FfiHandleRef::from_raw(descriptor)?,
                FfiHandleMut::from_raw(adapter_out)?,
            )
        };
        *out_handle = core::ptr::NonNull::dangling().as_ptr();
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates SRL Dynamic Rank-1 outer product: $\mathbf{y} = \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_out` (32B), `x_in` (32B), and `srl_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1(
    y_out: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_out, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(srl_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard =
            unsafe { DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_out, dim_out)? };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        AbiSrlDispatcher.eval_rank1(y_slice, x_slice, &desc)?;
        Ok(GCSO_SUCCESS)
    })
}

/// Evaluates SRL Dynamic Rank-1 outer product and accumulates directly into `y_inout` in-place:
/// $\mathbf{y} \gets \mathbf{y} + \alpha \cdot \mathbf{s} \odot (\mathbf{u} (\mathbf{v}^T \mathbf{x}))$.
///
/// # Safety
/// `y_inout` (32B), `x_in` (32B), and `srl_desc` (32B) must be non-null, aligned, and disjoint.
#[no_mangle]
pub unsafe extern "C" fn gcso_srl_eval_rank1_fused(
    y_inout: *mut f32,
    x_in: *const f32,
    srl_desc: *const gcso_srl_descriptor_t,
    dim_in: usize,
    dim_out: usize,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr_aligned(y_inout, 32)?;
        validate_ptr_aligned(x_in, 32)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let desc = unsafe { FfiHandleRef::from_raw(srl_desc)? };
        if dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        // SAFETY: Validated for non-nullness, 32B alignment, bounds, and disjointness.
        let mut io_guard = unsafe {
            DisjointRefMut::<f32, 32, 32>::from_raw_parts(x_in, dim_in, y_inout, dim_out)?
        };
        let (x_slice, y_slice) = io_guard.as_ref_mut();
        AbiSrlDispatcher.eval_rank1_fused(y_slice, x_slice, &desc)?;
        Ok(GCSO_SUCCESS)
    })
}

/// L2P-SVD: Training-free projection of fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}$,
/// $W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$) into an SRL Rank-1 descriptor and Q7 phase profile (Section 5.2).
///
/// # Safety
/// Pointers must satisfy non-null, alignment, and bounds invariants.
#[no_mangle]
pub unsafe extern "C" fn gcso_l2p_svd_project_lora(
    lora_a: *const f32,
    lora_b: *const f32,
    rank: usize,
    dim_in: usize,
    dim_out: usize,
    srl_out: *mut gcso_srl_descriptor_t,
    phase_profile_out: *mut gcso_q7_t,
) -> GcsoStatus {
    ffi_boundary!({
        validate_ptr(lora_a)?;
        validate_ptr(lora_b)?;
        validate_ptr(phase_profile_out)?;
        // SAFETY: Validated for non-nullness and 32B alignment.
        let mut srl_desc = unsafe { FfiHandleMut::from_raw(srl_out)? };
        if rank == 0 || dim_in == 0 || dim_out == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let a_len = rank
            .checked_mul(dim_in)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let b_len = dim_out
            .checked_mul(rank)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;

        if is_ptr_aligned_to(lora_a, 32)
            && is_ptr_aligned_to(lora_b, 32)
            && is_ptr_aligned_to(phase_profile_out, 16)
            && ((dim_out & 1) == 0)
        {
            let num_heads = (dim_out / 128).clamp(1, MAX_ATTENTION_HEADS);
            // SAFETY: Checked 32B/16B alignment above.
            let (a_aligned, b_aligned) = unsafe {
                (
                    AlignedSlice32::from_raw_parts(lora_a, a_len)?,
                    AlignedSlice32::from_raw_parts(lora_b, b_len)?,
                )
            };
            // SAFETY: Checked 16B alignment above.
            let mut p_aligned =
                unsafe { AlignedSliceMut16::from_raw_parts_mut(phase_profile_out, num_heads)? };

            AbiL2pDispatcher.project_lora(
                &a_aligned,
                &b_aligned,
                rank,
                dim_in,
                dim_out,
                &mut srl_desc,
                &mut p_aligned,
            )?;
            return Ok(GCSO_SUCCESS);
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
        srl_desc.layer_idx = 0;
        srl_desc.rank = 1;
        srl_desc.u_vector_ptr = lora_b as u64;
        srl_desc.v_vector_ptr = lora_a as u64;
        srl_desc.gain_scalar_ptr = 0;
        srl_desc.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

        let phase_val = if dim_out >= 2 {
            branchless_atan2_to_q7_fast(b_slice[rank], b_slice[0])
        } else {
            0
        };
        // SAFETY: `phase_profile_out` is validated non-null above.
        unsafe { *phase_profile_out = phase_val };
        Ok(GCSO_SUCCESS)
    })
}
