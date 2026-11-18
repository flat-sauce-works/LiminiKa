// name: src/core/src/phase_steering/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic Phase-Shifted RoPE (DPSR), RIPA soft-clamping, 8-lane SWAR QDPS, and
//! [`ManifoldGatewaySignal`] closed-loop phase steering.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Closed-Loop Steering)
//! - **[`PhaseSteeringEngine`]**: Compact 16-byte (`#[repr(C, align(16))]`) [`Pod`] state descriptor storing
//!   `(head_dim, num_heads, ripa_clamp_max_rad, qdps_min_step_rad)`. Delegates low-level
//!   constant-time AVX2/SSE2 SIMD and 8-lane SWAR operations directly to the default trait methods
//!   of [`PhaseSteering`], [`SubHeadRouter`], [`VisualAttentionSteering`], and
//!   [`crate::traits::BranchlessKernel`] (`crate::traits::rope`).
//! - **[`PhaseSteeringClosedLoop`]**: Blanket-implemented extension trait
//!   (`impl<T: PhaseSteering + SubHeadRouter + ?Sized> PhaseSteeringClosedLoop for T {}`) synthesizing
//!   any [`ManifoldGatewaySignal`] into a 128-byte [`GatewayFrame`] and [`CausalProvenanceReport`]
//!   while executing $SO(2)$-fused RIPA+sub-head router steering (`steer_gateway_signal_safe`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]

use core::mem::{align_of, offset_of, size_of};

use crate::abi::{gcso_config_t, gcso_pointer_trail_t, GCSO_ERROR_INVALID_ARGUMENT};
use crate::traits::aligned::AlignedSliceMut32;
use crate::traits::gateway::{CausalProvenanceReport, GatewayFrame, ManifoldGatewaySignal};
use crate::traits::primitives::{
    branchless_is_finite_f32, branchless_min_f32, branchless_select_f32, branchless_select_u32,
};
use crate::traits::primitives::{
    GcsoResult, Pod, DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, MAX_ATTENTION_HEADS,
    Q7_PHASE_SCALE_RAD,
};
use crate::traits::rope::{PhaseSteering, SubHeadRouter, VisualAttentionSteering};
use crate::traits::telemetry::{
    HEADS_PER_STIGMERGIC_GRANULE, MAX_STIGMERGIC_DENSITY, NUM_STIGMERGIC_GRANULES,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
/// Default attention head dimension ($d_{\mathrm{head}} = 128$) for [`PhaseSteeringEngine::default`].
pub const DEFAULT_DPSR_HEAD_DIM: u32 = 128;

/// Default attention head count ($H = 32$) for [`PhaseSteeringEngine::default`].
pub const DEFAULT_DPSR_NUM_HEADS: u32 = 32;

/// Scale factor for Q7 quantized phase representation ($\beta_{\mathrm{Q7}} = \pi / 128$).
pub const Q7_PHASE_SCALE: f32 = Q7_PHASE_SCALE_RAD;

// ===================================================================
// Internal Constant-Time Register-Resident SWAR & SO(2) Fusion Kernels
// ===================================================================

/// Validates `(head_dim, num_heads)`, `max_rad > 0.0`, and `query_tensor` / `key_tensor` slice
/// bounds and `GCSO_RESTRICT` non-aliasing invariants for closed-loop steering methods.
#[inline(always)]
pub(super) fn validate_closed_loop_tensors<T: PhaseSteering + ?Sized>(
    engine: &T,
    max_rad: f32,
    query_tensor: &AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(usize, usize)> {
    if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let (head_dim, num_heads, expected_len) = engine.validate_head_layout()?;
    if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    if let Some(k) = key_tensor {
        if k.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(k)?;
    }
    Ok((head_dim, num_heads))
}

/// Internal register-resident Gateway steering core returning
/// `(GatewayFrame, CausalProvenanceReport, frame_words_64)`.
#[inline(always)]
pub(super) fn steer_gateway_signal_core_safe<
    T: PhaseSteering + SubHeadRouter + ?Sized,
    S: ManifoldGatewaySignal + ?Sized,
>(
    engine: &T,
    signal: &S,
    trail: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> GcsoResult<(
    GatewayFrame,
    CausalProvenanceReport,
    [u64; NUM_STIGMERGIC_GRANULES],
)> {
    let (frame, frm_w) = signal.synthesize_gateway_frame_with_words();
    let report = frame.to_causal_report();
    let (head_dim, num_heads) = validate_closed_loop_tensors(
        engine,
        frame.ripa_clamp_rad,
        query_tensor,
        key_tensor.as_deref(),
    )?;

    trail.prev_ptr = trail.current_ptr;
    trail.current_ptr = frame.tagged_ptr;
    let has_frame_anchor = frame.target_anchor_id != 0;
    trail.target_anchor_id = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        trail.target_anchor_id,
    );
    trail.cluster_id = branchless_select_u32(
        has_frame_anchor,
        frame.target_anchor_id,
        frame.dominant_slot as u32,
    );

    trail.stigmergic_density =
        branchless_min_f32(trail.stigmergic_density + 0.25, MAX_STIGMERGIC_DENSITY);
    trail.step_count = trail.step_count.wrapping_add(1);

    let is_repulsed = frame.is_repulsed();
    let rep_flags = trail.flags | TRAIL_FLAG_REPULSION_ACTIVE;
    let coh_flags = (trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
        & !TRAIL_FLAG_REPULSION_ACTIVE;
    trail.flags = branchless_select_u32(is_repulsed, rep_flags, coh_flags);

    let _ = crate::traits::gateway::accumulate_trail_and_steer_gateway_words_core(
        &frame,
        &frm_w,
        trail,
        unresolved_damping_u7,
        head_dim,
        num_heads,
        Some(query_tensor),
        key_tensor,
        logits_opt,
    )?;

    Ok((frame, report, frm_w))
}

pub mod closed_loop;
pub mod engine;

pub use self::{closed_loop::*, engine::*};

#[cfg(test)]
mod tests;
