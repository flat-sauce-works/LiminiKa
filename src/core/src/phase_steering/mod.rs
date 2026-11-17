// name: src/core/src/phase_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic Phase-Shifted RoPE (DPSR), RIPA soft-clamping, 8-lane SWAR QDPS, Single Unified
//! Attractor Manifold steering, Bayesian Pointer-Trail Attention modulation, Limina
//! inter-memory transition engine, Online Toroidal $k$-Means / DP-Means Coarse-to-Fine
//! routing, and 2-Origin Composite Manifold Gateway execution.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Closed-Loop Steering)
//! - **[`PhaseSteeringEngine`]**: Compact 16-byte (`#[repr(C, align(16))]`) [`Pod`] state descriptor storing
//!   `(head_dim, num_heads, ripa_clamp_max_rad, qdps_min_step_rad)`. Delegates all 50+ low-level
//!   constant-time AVX2/SSE2 SIMD and 8-lane SWAR operations directly to the default trait methods
//!   of [`PhaseSteering`], [`EntropyGatedPhaseLookup`], [`DynamicLayerSkipper`], [`SubHeadRouter`],
//!   [`VisualAttentionSteering`], and [`crate::traits::BranchlessKernel`] (`crate::traits::rope`).
//! - **[`PhaseSteeringClosedLoop`]**: Blanket-implemented extension trait
//!   (`impl<T: PhaseSteering + SubHeadRouter + ?Sized> PhaseSteeringClosedLoop for T {}`) that equips
//!   [`PhaseSteeringEngine`], [`crate::traits::ActionHub`], [`crate::traits::HotPathObserver`], and
//!   [`crate::traits::ColdPathObserver`] with:
//!   1. Closed-loop 8-cell SWAR Micro-Swarm resolution (`resolve_and_apply_swarm_phase_steering_safe`,
//!      `resolve_unified_field_swarm_and_steer_safe`, `resolve_slot_bank_swarm_and_steer_safe`)
//!      operating directly on register-resident `[u64; 8]` SWAR words (`*_to_words`) with
//!      single-scan $\mathcal{O}(1)$ dominant-donor recruitment, zero-stack-spill
//!      `Q7_GEODESIC_DEFAULT_RIPA_SINCOS_LUT_256` lookup (`steer_from_swar_words_core_safe`), and
//!      zero redundant memory reload/store round-trips.
//!   2. Zero-copy candidate-trail Bayes-Attention score modulation
//!      (`modulate_candidate_trail_attention_scores_zero_copy`).
//!   3. Single-pass fused **Autonomous Stigmergic Pointer-Trail selection, rewiring, Limina bridging,
//!      in-place score modulation, and register-resident RIPA steering**
//!      (`select_rewire_and_steer_bayesian_pointer_trail_safe`), combining sequential conditional
//!      transition probability ($\prod_t P(M_t \mid M_{<t})$ $\times$ Attention phase interference)
//!      with indirect pointer-chain routing in `ActionHub`.
//!   4. Register-resident $SO(2)$ group-closure fused **Unified-Field RIPA + sub-head router Sub-Head Pseudo-MoE
//!      routing + multiplicative inverse-temperature logit sharpening/flattening**
//!      (`steer_unified_field_pspm_and_inv_temperature_safe`) with zero intermediate stack buffers.
//!   5. Orthogonal Procrustes trail feedback (`steer_trail_with_procrustes_feedback`) and Limina
//!      inter-memory bridging (`bridge_liminal_trail_and_steer_qk_safe`).
//!   6. **2-Origin Composite Manifold Gateway, Online Toroidal $k$-Means / DP-Means Macro Clustering,
//!      Coarse-to-Fine $\mathcal{O}(K) \to \mathcal{O}(M_{\mathrm{local}})$ Routing & `.gcso`
//!      Crystallization** (`steer_gateway_signal_safe`, `steer_and_crystallize_gateway_signal_safe`,
//!      `steer_and_kmeans_crystallize_gateway_signal_safe`,
//!      `route_coarse_to_fine_and_steer_bayesian_trails_safe`,
//!      `steer_crystallized_ensemble_zero_copy_safe`,
//!      `steer_gateway_swarm_bayesian_and_crystallize_safe`), synthesizing any [`ManifoldGatewaySignal`]
//!      into a 128-byte [`GatewayFrame`] and [`CausalProvenanceReport`] while executing $SO(2)$-fused
//!      RIPA+sub-head router steering and accumulating macro centroid distributions on $SO(2)^{64}$.

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

use crate::abi::{
    gcso_config_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_subhead_config_t,
    gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::aligned::{AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut32};
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_phase_coherence_inv_temp_scale,
    coherence_from_word_arrays_with_both_uniform_flags_64, procrustes_q7_words_64_simd,
    scale_f32_slice_unrolled8, swar_words_equal_64, swar_words_nonzero_64,
};
use crate::traits::gateway::{
    CausalProvenanceReport, CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
};
use crate::traits::primitives::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32,
    branchless_popcount_rsqrt_u32, branchless_rad_to_qdps_step_u8, branchless_relu_f32,
    branchless_rsqrt_f32, branchless_select_f32, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, fast_inv_len_f32, is_word_array_uniform_64, load_q7_words_64,
    swar_q7_circular_add_x8, swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64,
    swar_q7_qdps_filter_x8, swar_q7_scale_le128_with_bias_x8, trail_phase_words_mut,
    trail_phase_words_ref, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    GcsoResult, Pod, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_SUBHEAD_ROUTE,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_REPULSION, DEFAULT_ENTROPY_EPS,
    DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, ripa_upper_rot_dim, steer_from_swar_words_core_safe,
    DynamicLayerSkipper, EntropyGatedPhaseLookup, PhaseSteering, SubHeadRouter,
    VisualAttentionSteering,
};
use crate::traits::telemetry::{
    select_and_bridge_macro_aligned_bayesian_trail_core, ContrastivePhaseBloomFilter,
    PagedBitmaskOps, StigmergicActionPtrOps, StigmergicSwarmOps, StigmergicTrailOps,
    SwarmResolutionReport, HEADS_PER_SWARM_CELL, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
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

/// Zero-cost transmutation of 8 register-resident `u64` SWAR words (`[u64; 8]`) into a
/// 64-head Q7 phase array (`[gcso_q7_t; 64]`) on little-endian targets.
#[inline(always)]
#[must_use]
pub(super) const fn words_to_q7_array_64(
    words: &[u64; NUM_SWARM_MICRO_CELLS],
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    const _: () = assert!(cfg!(target_endian = "little"));
    // SAFETY: `[u64; 8]` and `[gcso_q7_t; 64]` have identical 64-byte size and Pod bit validity.
    unsafe {
        core::mem::transmute::<[u64; NUM_SWARM_MICRO_CELLS], [gcso_q7_t; MAX_ATTENTION_HEADS]>(
            *words,
        )
    }
}

/// Filters 8 register-resident `u64` SWAR words (`raw_words`) via 8-lane SWAR QDPS (`min_step_rad`)
/// and steers `query_tensor` (and optional `key_tensor`) via `steer_from_swar_words_core_safe`
/// without reloading `trail.accumulated_phase_delta` from memory.
#[inline(always)]
pub(super) fn filter_qdps_and_steer_from_words_64(
    raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
    min_step_rad: f32,
    max_rad: f32,
    head_dim: usize,
    num_heads: usize,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
    if step_u8 <= 1 {
        if (raw_words[0] == 0) && !swar_words_nonzero_64(raw_words) {
            return;
        }
        steer_from_swar_words_core_safe(
            raw_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            false,
            max_rad,
            query_tensor,
            key_tensor,
        );
    } else {
        let swar_words = swar_q7_qdps_filter_words_64(raw_words, step_u8);
        if (swar_words[0] == 0) && !swar_words_nonzero_64(&swar_words) {
            return;
        }
        steer_from_swar_words_core_safe(
            &swar_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            false,
            max_rad,
            query_tensor,
            key_tensor,
        );
    }
}

#[inline(always)]
pub(super) fn filter_qdps_and_steer_from_phases_64(
    raw_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    min_step_rad: f32,
    max_rad: f32,
    head_dim: usize,
    num_heads: usize,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    let raw_words = load_q7_words_64(raw_phases);
    filter_qdps_and_steer_from_words_64(
        &raw_words,
        min_step_rad,
        max_rad,
        head_dim,
        num_heads,
        query_tensor,
        key_tensor,
    );
}

/// Steers `query_tensor` (and optional `key_tensor`) directly from the 8 register-resident `u64`
/// SWAR words (`swar_words`) returned by `StigmergicSwarmOps::*_to_words`, eliminating the
/// redundant memory reload of `trail.accumulated_phase_delta` and double QDPS filtering when
/// `qdps_min_step_q7 <= 1`.
#[inline(always)]
pub(super) fn finish_constant_time_swarm_qdps_and_steer(
    evolved_ptr: gcso_tagged_ptr_t,
    quorum_mask: u8,
    swar_words: &[u64; NUM_SWARM_MICRO_CELLS],
    qdps_min_step_q7: u8,
    unresolved_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    max_rad: f32,
    trail: &mut gcso_pointer_trail_t,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    let eff_words = if qdps_min_step_q7 <= 1 {
        *swar_words
    } else {
        let filt_w = swar_q7_qdps_filter_words_64(swar_words, qdps_min_step_q7);
        *trail_phase_words_mut(trail) = filt_w;
        filt_w
    };

    if !evolved_ptr.has_action_mode(ACTION_MODE_DPSR_STEER) || !swar_words_nonzero_64(&eff_words) {
        return;
    }

    let clean_damping =
        branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
    steer_from_swar_words_core_safe(
        &eff_words,
        quorum_mask,
        clean_damping,
        head_dim,
        num_heads,
        true,
        max_rad,
        query_tensor,
        key_tensor,
    );
}

/// Internal register-resident Gateway steering core that returns
/// `(GatewayFrame, CausalProvenanceReport, frame_words_64)` so downstream crystallization and
/// Online Toroidal $k$-Means / DP-Means methods can pass `frame_words_64` directly to
/// [`CrystallizedAttractorRecord::absorb_gateway_frame_with_words`] without reloading `frame.phase_deltas`.
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
    [u64; NUM_SWARM_MICRO_CELLS],
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

/// Register-resident Fine-Step candidate scanner, fallback scanner, Limina geodesic SLERP bridger,
/// and QDPS/RIPA steerer shared by [`PhaseSteeringClosedLoop::route_coarse_to_fine_and_steer_bayesian_trails_safe`]
/// and [`PhaseSteeringClosedLoop::route_top2_liminal_and_steer_bayesian_trails_safe`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn route_macro_aligned_bayesian_trails_and_steer_core(
    self_w: [u64; NUM_SWARM_MICRO_CELLS],
    win_w: &[u64; NUM_SWARM_MICRO_CELLS],
    has_win_centroid: bool,
    primary_macro_anchor_id: u32,
    secondary_macro_anchor_id: u32,
    src_trail: &mut gcso_pointer_trail_t,
    candidate_trails: &mut [gcso_pointer_trail_t],
    head_dim: usize,
    num_heads: usize,
    phase_interference_gain: f32,
    stigmergic_prior_gain: f32,
    pointer_link_bonus: f32,
    repulsion_penalty: f32,
    fine_interpolation_u7: u8,
    contradiction_threshold: f32,
    min_step_rad: f32,
    max_rad: f32,
    attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
) -> (u32, f32, f32, f32, bool) {
    let outcome = select_and_bridge_macro_aligned_bayesian_trail_core(
        self_w,
        if has_win_centroid { Some(win_w) } else { None },
        primary_macro_anchor_id,
        secondary_macro_anchor_id,
        src_trail,
        candidate_trails,
        attention_scores.map(AlignedSliceMut32::as_mut_slice),
        head_dim as u32,
        phase_interference_gain,
        stigmergic_prior_gain,
        pointer_link_bonus,
        repulsion_penalty,
        fine_interpolation_u7,
        contradiction_threshold,
        min_step_rad,
        true,
    );

    let mut steered_w = outcome.self_words;
    let (coh_out, trans_lp) = if outcome.found_valid {
        let dst_trail = &mut candidate_trails[outcome.best_idx];
        dst_trail.prev_ptr = src_trail.current_ptr;
        dst_trail.step_count = src_trail.step_count.wrapping_add(1);
        let inherit_anchor = dst_trail.target_anchor_id != 0;
        src_trail.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            dst_trail.target_anchor_id,
            src_trail.target_anchor_id,
        );
        let resolved_cluster_id = branchless_select_u32(
            dst_trail.cluster_id != 0,
            dst_trail.cluster_id,
            branchless_select_u32(
                (primary_macro_anchor_id != 0) & (src_trail.cluster_id != 0),
                src_trail.cluster_id,
                src_trail.target_anchor_id,
            ),
        );
        src_trail.cluster_id = resolved_cluster_id;
        dst_trail.target_anchor_id = src_trail.target_anchor_id;
        dst_trail.cluster_id = resolved_cluster_id;

        (
            outcome.best_coh,
            branchless_fast_transition_log_prob(outcome.best_coh),
        )
    } else {
        steered_w = [
            swar_q7_neg_sat_x8(steered_w[0]),
            swar_q7_neg_sat_x8(steered_w[1]),
            swar_q7_neg_sat_x8(steered_w[2]),
            swar_q7_neg_sat_x8(steered_w[3]),
            swar_q7_neg_sat_x8(steered_w[4]),
            swar_q7_neg_sat_x8(steered_w[5]),
            swar_q7_neg_sat_x8(steered_w[6]),
            swar_q7_neg_sat_x8(steered_w[7]),
        ];
        *trail_phase_words_mut(src_trail) = steered_w;
        src_trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        let cur_ptr = gcso_tagged_ptr_t(src_trail.current_ptr);
        let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
        src_trail.current_ptr = cur_ptr
            .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
            .conjugate_phase()
            .raw();
        src_trail.cluster_id = ATTRACTOR_SLOT_REPULSION as u32;
        let mean_coh = if outcome.coh_cnt > 0 {
            branchless_clamp_f32(
                outcome.coh_sum * fast_inv_len_f32(outcome.coh_cnt),
                -1.0,
                1.0,
            )
        } else {
            0.0_f32
        };
        (mean_coh, 0.0_f32)
    };

    filter_qdps_and_steer_from_words_64(
        &steered_w,
        branchless_select_f32(outcome.qdps_done, 0.0, min_step_rad),
        max_rad,
        head_dim,
        num_heads,
        query_tensor,
        key_tensor,
    );

    let out_score = branchless_select_f32(outcome.found_valid, outcome.best_score, 0.0);
    (
        outcome.best_id,
        out_score,
        coh_out,
        trans_lp,
        outcome.found_valid,
    )
}

pub mod closed_loop;
pub mod engine;

pub use self::{closed_loop::*, engine::*};

#[cfg(test)]
mod tests;
