// name: src/core/src/dpsr.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Dynamic Phase-Shifted RoPE (DPSR), RIPA soft-clamping, 8-lane SWAR QDPS, Single Unified
//! Attractor Manifold steering, Bayesian Pointer-Trail Attention modulation, Limina
//! inter-memory transition engine, Online Toroidal $k$-Means / DP-Means Coarse-to-Fine
//! routing, and 2-Origin Composite Manifold Gateway execution.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Closed-Loop Steering)
//! - **[`DpsrEngine`]**: Compact 16-byte (`#[repr(C, align(16))]`) [`Pod`] state descriptor storing
//!   `(head_dim, num_heads, ripa_clamp_max_rad, qdps_min_step_rad)`. Delegates all 50+ low-level
//!   constant-time AVX2/SSE2 SIMD and 8-lane SWAR operations directly to the default trait methods
//!   of [`PhaseSteering`], [`EntropyGatedPhaseLookup`], [`DynamicLayerSkipper`], [`PspmRouter`],
//!   [`VisualAttentionSteering`], and [`crate::traits::BranchlessKernel`] (`crate::traits::rope`).
//! - **[`DpsrClosedLoopSteering`]**: Blanket-implemented extension trait
//!   (`impl<T: PhaseSteering + PspmRouter + ?Sized> DpsrClosedLoopSteering for T {}`) that equips
//!   [`DpsrEngine`], [`crate::traits::ActionHub`], [`crate::traits::HotPathObserver`], and
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
//!      (`select_rewire_and_steer_bayesian_pointer_trail_safe`), uniting Hypothesis 1 (Persona as
//!      Joint Probability Chain Rule $\times$ Attention phase interference) and Hypothesis 2
//!      (Intelligence as Stigmergic Pointer-Trail manipulation in `ActionHub`).
//!   4. Register-resident $SO(2)$ group-closure fused **Unified-Field RIPA + PSPM Sub-Head Pseudo-MoE
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
//!      RIPA+PSPM steering and accumulating macro centroid distributions on $SO(2)^{64}$.

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
    gcso_config_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t,
    gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::aligned::{AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut32};
use crate::traits::branchless::{
    branchless_abs_f32, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_ln_f32, branchless_max_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_popcount_rsqrt_u32,
    branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_rsqrt_f32,
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    coherence_from_word_arrays_with_both_uniform_flags_64, fast_inv_len_f32,
    fast_ln_1p_bounded_f32, is_word_array_uniform_64, load_q7_words_64,
    procrustes_q7_words_64_simd, scale_f32_slice_unrolled8, swar_q7_circular_add_x8,
    swar_q7_neg_sat_x8, swar_q7_qdps_filter_x8, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_uniform_flag_64, trail_phase_words_mut, trail_phase_words_ref,
    Q7_FULL_COS_REF,
};
use crate::traits::gateway::{
    CausalProvenanceReport, CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal,
    DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
};
use crate::traits::primitives::{
    GcsoResult, Pod, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PSPM_ROUTE,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_REPULSION, DEFAULT_ENTROPY_EPS,
    DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, MAX_ATTENTION_HEADS,
    MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use crate::traits::rope::{
    dispatch_fused_ripa_and_pspm_words_core, steer_from_swar_words_core_safe, DynamicLayerSkipper,
    EntropyGatedPhaseLookup, PhaseSteering, PspmRouter, VisualAttentionSteering,
};
use crate::traits::telemetry::{
    ContrastivePhaseBloomFilter, PagedBitmaskOps, StigmergicActionPtrOps, StigmergicSwarmOps,
    StigmergicTrailOps, SwarmResolutionReport, HEADS_PER_SWARM_CELL, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use crate::traits::{
    filter_qdps_words_in_registers_64, swar_words_equal_64, swar_words_nonzero_64,
};

/// Default attention head dimension ($d_{\mathrm{head}} = 128$) for [`DpsrEngine::default`].
pub const DEFAULT_DPSR_HEAD_DIM: u32 = 128;

/// Default attention head count ($H = 32$) for [`DpsrEngine::default`].
pub const DEFAULT_DPSR_NUM_HEADS: u32 = 32;

/// Scale factor for Q7 quantized phase representation ($\beta_{\mathrm{Q7}} = \pi / 128$).
pub const Q7_PHASE_SCALE: f32 = Q7_PHASE_SCALE_RAD;

/// 64-bit SWAR byte-splat multiplier (`0x0101_0101_0101_0101`).
const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;

/// Evaluates the conditional transition log-probability $\ln P(M_t \mid M_{t-1}) = \ln(0.5(1 + c))$
/// using the 1-lookup identity $\ln(1 + c) - \ln 2$ (`fast_ln_1p_bounded_f32`, `0` divisions)
/// whenever $c \ge 0$, falling back to [`branchless_ln_f32`] only when $c < 0$.
#[inline(always)]
#[must_use]
const fn fast_transition_log_prob_from_coh(coh: f32) -> f32 {
    if coh >= 0.999_999 {
        0.0_f32
    } else if coh >= 0.0 {
        branchless_flush_subnormal_f32(fast_ln_1p_bounded_f32(coh) - core::f32::consts::LN_2)
    } else {
        let cond_prob = branchless_clamp_f32(0.5 * (1.0 + coh), DEFAULT_ENTROPY_EPS, 1.0);
        branchless_flush_subnormal_f32(branchless_ln_f32(cond_prob))
    }
}

// ===================================================================
// Internal Constant-Time Register-Resident SWAR & SO(2) Fusion Kernels
// ===================================================================

/// Validates `(head_dim, num_heads)`, `max_rad > 0.0`, and `query_tensor` / `key_tensor` slice
/// bounds and `GCSO_RESTRICT` non-aliasing invariants for closed-loop steering methods.
#[inline(always)]
fn validate_closed_loop_tensors<T: PhaseSteering + ?Sized>(
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
const fn words_to_q7_array_64(
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

/// Filters 8 register-resident `u64` SWAR words (`raw_words`) via 8-lane SWAR QDPS (`step_u8`),
/// bypassing the 8 filter calls when `step_u8 <= 1` (since every non-zero `i8` already satisfies
/// `|x| >= 1`).
#[inline(always)]
#[must_use]
const fn filter_qdps_words_64(
    raw_words: &[u64; NUM_SWARM_MICRO_CELLS],
    step_u8: u8,
) -> [u64; NUM_SWARM_MICRO_CELLS] {
    if step_u8 <= 1 {
        *raw_words
    } else {
        filter_qdps_words_in_registers_64(raw_words, step_u8)
    }
}

/// Filters 8 register-resident `u64` SWAR words (`raw_words`) via 8-lane SWAR QDPS (`min_step_rad`)
/// and steers `query_tensor` (and optional `key_tensor`) via `steer_from_swar_words_core_safe`
/// without reloading `trail.accumulated_phase_delta` from memory.
#[inline(always)]
fn filter_qdps_and_steer_from_words_64(
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
        let swar_words = filter_qdps_words_in_registers_64(raw_words, step_u8);
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
fn filter_qdps_and_steer_from_phases_64(
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
fn finish_constant_time_swarm_qdps_and_steer(
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
        let filt_w = filter_qdps_words_64(swar_words, qdps_min_step_q7);
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
fn steer_gateway_signal_core_safe<
    T: PhaseSteering + PspmRouter + ?Sized,
    S: ManifoldGatewaySignal + ?Sized,
>(
    engine: &T,
    signal: &S,
    trail: &mut gcso_pointer_trail_t,
    unresolved_damping_u7: u8,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
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

    let frm_nonzero = swar_words_nonzero_64(&frm_w);
    let next_w = if frm_nonzero {
        *trail_phase_words_mut(trail) = frm_w;
        frm_w
    } else {
        *trail_phase_words_ref(trail)
    };

    trail.stigmergic_density =
        branchless_min_f32(trail.stigmergic_density + 0.25, MAX_STIGMERGIC_DENSITY);
    trail.attractor_pull_force = branchless_min_f32(
        branchless_max_f32(trail.attractor_pull_force, frame.effective_energy),
        MAX_ATTRACTOR_PULL_FORCE,
    );
    trail.step_count = trail.step_count.wrapping_add(1);

    let is_repulsed = frame.is_repulsed();
    let rep_flags = trail.flags | TRAIL_FLAG_REPULSION_ACTIVE;
    let coh_flags = (trail.flags | TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE)
        & !TRAIL_FLAG_REPULSION_ACTIVE;
    trail.flags = branchless_select_u32(is_repulsed, rep_flags, coh_flags);

    if (frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) == 0 {
        let swar_words = if frm_nonzero | (frame.qdps_min_step_q7 <= 1) {
            next_w
        } else {
            filter_qdps_words_64(&next_w, frame.qdps_min_step_q7)
        };
        let swar_nonzero = if frm_nonzero {
            true
        } else {
            swar_words_nonzero_64(&swar_words)
        };
        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
        let use_pspm = ((frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0)
            & (frame.active_swarm_cell_mask == 0xFF);
        if use_pspm {
            if swar_nonzero {
                let is_default_ripa = (frame.ripa_clamp_rad.to_bits()
                    == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                    & (num_heads > 0)
                    & (num_heads <= MAX_ATTENTION_HEADS);
                if is_default_ripa {
                    let rot_dim = ((head_dim >> 2) & !1).max(2).min(head_dim);
                    if let Some(k) = key_tensor.as_deref_mut() {
                        crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            query_tensor.as_mut_slice(),
                            k.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                        );
                    } else {
                        crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                            query_tensor.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                            true,
                        );
                    }
                } else {
                    let pspm_cfg = frame.to_pspm_config_for_heads(num_heads);
                    dispatch_fused_ripa_and_pspm_words_core(
                        query_tensor,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        &pspm_cfg,
                        &swar_words,
                        &swar_words,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        0,
                        head_dim,
                        true,
                    )?;
                    if let Some(k) = key_tensor.as_deref_mut() {
                        steer_from_swar_words_core_safe(
                            &swar_words,
                            frame.active_swarm_cell_mask,
                            clean_damping,
                            head_dim,
                            num_heads,
                            false,
                            frame.ripa_clamp_rad,
                            k,
                            None,
                        );
                    }
                }
            }
        } else if ((frame.action_mode_flags & ACTION_MODE_DPSR_STEER) != 0) & swar_nonzero {
            steer_from_swar_words_core_safe(
                &swar_words,
                frame.active_swarm_cell_mask,
                clean_damping,
                head_dim,
                num_heads,
                false,
                frame.ripa_clamp_rad,
                query_tensor,
                key_tensor,
            );
        }
    }

    if let Some(logits) = logits_opt {
        if !logits.is_empty() {
            scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
        }
    }

    Ok((frame, report, frm_w))
}

/// Register-resident Fine-Step candidate scanner, fallback scanner, Limina geodesic SLERP bridger,
/// and QDPS/RIPA steerer shared by [`DpsrClosedLoopSteering::route_coarse_to_fine_and_steer_bayesian_trails_safe`]
/// and [`DpsrClosedLoopSteering::route_top2_liminal_and_steer_bayesian_trails_safe`].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn route_macro_aligned_bayesian_trails_and_steer_core(
    mut self_w: [u64; NUM_SWARM_MICRO_CELLS],
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
) -> (u32, f32, f32, bool) {
    let cand_len = candidate_trails.len();
    let safe_dim = branchless_select_u32(head_dim > 0, head_dim as u32, 128);
    let rsqrt_d = if safe_dim <= 256 {
        branchless_popcount_rsqrt_u32(safe_dim)
    } else {
        branchless_rsqrt_f32(safe_dim as f32)
    };
    let phase_scale =
        branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain) * rsqrt_d);
    let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
    let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
    let safe_thresh = branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
    let clean_penalty = branchless_flush_subnormal_f32(branchless_relu_f32(repulsion_penalty));
    let cur_linked_id = src_trail.linked_trail_id;
    let self_is_uniform = is_word_array_uniform_64(&self_w);
    let self_w0 = self_w[0];
    let self_b0 = (self_w0 & 0xFF) as u8;
    let self_is_byte_uniform = self_is_uniform & (self_w0 == self_w0.rotate_right(8));
    let mut scores_slice_opt = attention_scores.map(AlignedSliceMut32::as_mut_slice);
    let prune_non_macro = primary_macro_anchor_id != 0;

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn fallback_scan_unpruned_candidates_dpsr_cold(
        self_w: &[u64; NUM_SWARM_MICRO_CELLS],
        self_is_uniform: bool,
        self_is_byte_uniform: bool,
        self_b0: u8,
        candidate_trails: &[gcso_pointer_trail_t],
        primary_macro_anchor_id: u32,
        secondary_macro_anchor_id: u32,
        cur_linked_id: u32,
        phase_scale: f32,
        clean_eta: f32,
        safe_thresh: f32,
        best_id: &mut u32,
        best_idx: &mut usize,
        best_score: &mut f32,
        best_coh: &mut f32,
        best_link_term: &mut f32,
        best_cand_is_uniform: &mut bool,
        found: &mut bool,
    ) {
        let cand_len = candidate_trails.len();
        let mut f_idx = 0usize;
        while f_idx < cand_len {
            let cand = &candidate_trails[f_idx];
            let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
            if !cand_not_repulsed {
                f_idx += 1;
                continue;
            }
            let cand_id = f_idx as u32;
            let in_c1 = (primary_macro_anchor_id != 0)
                & ((cand.cluster_id == primary_macro_anchor_id)
                    | (cand.target_anchor_id == primary_macro_anchor_id));
            let in_c2 = (secondary_macro_anchor_id != 0)
                & ((cand.cluster_id == secondary_macro_anchor_id)
                    | (cand.target_anchor_id == secondary_macro_anchor_id));
            let was_checked = in_c1
                | in_c2
                | (cand.cluster_id == 0)
                | ((cur_linked_id != 0) & (cur_linked_id == cand_id));
            if was_checked {
                f_idx += 1;
                continue;
            }
            let cand_w = trail_phase_words_ref(cand);
            let cu = self_is_uniform && is_word_array_uniform_64(cand_w);
            let coh = if cu {
                let cw0 = cand_w[0];
                if cw0 == self_w[0] {
                    1.0
                } else if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        self_w, true, cand_w, true,
                    )
                }
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    self_w,
                    self_is_uniform,
                    cand_w,
                    false,
                )
            };
            let prior_term = clean_eta * fast_ln_1p_bounded_f32(cand.stigmergic_density);
            let is_coherent = coh >= -safe_thresh;
            let bayes_delta = branchless_flush_subnormal_f32(
                phase_scale * coh + branchless_select_f32(is_coherent, prior_term, 0.0),
            );
            let take = is_coherent & ((!*found) | (bayes_delta > *best_score));
            if take {
                *best_id = cand_id;
                *best_idx = f_idx;
                *best_score = bayes_delta;
                *best_coh = coh;
                *best_link_term = 0.0;
                *best_cand_is_uniform = cu;
            }
            *found |= is_coherent;
            f_idx += 1;
        }
    }

    let mut best_id = cur_linked_id;
    let mut best_idx = 0usize;
    let mut best_score = -1.0e30_f32;
    let mut best_coh = 0.0_f32;
    let mut best_link_term = 0.0_f32;
    let mut best_cand_is_uniform = false;
    let mut best_cand_eq_self = false;
    let mut found = false;
    let mut win_meta_evaluated = false;
    let mut win_eq_self = false;
    let mut win_is_uniform = false;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (vs0, vs1) = unsafe {
        crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform)
    };
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    let (mut vw0, mut vw1) = (vs0, vs1);

    let mut idx = 0usize;
    while idx < cand_len {
        let cand = &candidate_trails[idx];
        let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        if (!cand_not_repulsed) & scores_slice_opt.is_none() {
            idx += 1;
            continue;
        }
        let cand_id = idx as u32;
        let in_c1 = (primary_macro_anchor_id != 0)
            & ((cand.cluster_id == primary_macro_anchor_id)
                | (cand.target_anchor_id == primary_macro_anchor_id));
        let in_c2 = (secondary_macro_anchor_id != 0)
            & ((cand.cluster_id == secondary_macro_anchor_id)
                | (cand.target_anchor_id == secondary_macro_anchor_id));
        let in_macro_cluster = in_c1 | in_c2;
        let is_unstamped = cand.cluster_id == 0;
        let is_cur_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
        if prune_non_macro & (!in_macro_cluster) & (!is_unstamped) & (!is_cur_linked) {
            idx += 1;
            continue;
        }

        let cand_w = trail_phase_words_ref(cand);
        let cw0 = cand_w[0];
        let cand_eq_self = (cw0 == self_w0) && swar_words_equal_64(&self_w, cand_w);
        let cand_is_uniform = if cand_eq_self {
            self_is_uniform
        } else {
            self_is_uniform && is_word_array_uniform_64(cand_w)
        };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut cand_vc = (vs0, vs1);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut cand_vc_loaded = false;
        let coh = if cand_eq_self {
            1.0
        } else if cand_is_uniform {
            if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
            } else {
                coherence_from_word_arrays_with_both_uniform_flags_64(&self_w, true, cand_w, true)
            }
        } else {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                unsafe {
                    use core::arch::x86_64::_mm256_sub_epi8;
                    let (vc0, vc1) = crate::traits::branchless::avx2_words64_to_vec64(cand_w);
                    cand_vc = (vc0, vc1);
                    cand_vc_loaded = true;
                    crate::traits::branchless::avx2_coherence_from_diff_vec64(
                        _mm256_sub_epi8(vs0, vc0),
                        _mm256_sub_epi8(vs1, vc1),
                    )
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &self_w,
                    self_is_uniform,
                    cand_w,
                    false,
                )
            }
        };
        let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
        let mut link_term = 0.0_f32;
        let eff_prior_link = if is_coherent {
            let prior_term = clean_eta * fast_ln_1p_bounded_f32(cand.stigmergic_density);
            let is_linked = is_cur_linked | in_macro_cluster;
            link_term = if is_linked {
                clean_link
            } else if has_win_centroid & is_unstamped {
                if !win_meta_evaluated {
                    win_eq_self = swar_words_equal_64(&self_w, win_w);
                    if !win_eq_self {
                        win_is_uniform = is_word_array_uniform_64(win_w);
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        unsafe {
                            let (w0, w1) =
                                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                    win_w,
                                    win_is_uniform,
                                );
                            vw0 = w0;
                            vw1 = w1;
                        }
                    }
                    win_meta_evaluated = true;
                }
                let win_coh = if win_eq_self {
                    coh
                } else if cand_is_uniform && win_is_uniform {
                    coherence_from_word_arrays_with_both_uniform_flags_64(win_w, true, cand_w, true)
                } else {
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    {
                        unsafe {
                            use core::arch::x86_64::_mm256_sub_epi8;
                            let (vc0, vc1) = if cand_vc_loaded {
                                cand_vc
                            } else {
                                let (v0, v1) =
                                    crate::traits::branchless::avx2_words64_to_vec64(cand_w);
                                cand_vc = (v0, v1);
                                cand_vc_loaded = true;
                                (v0, v1)
                            };
                            crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                _mm256_sub_epi8(vw0, vc0),
                                _mm256_sub_epi8(vw1, vc1),
                            )
                        }
                    }
                    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                    {
                        let cand_uni_for_win = if self_is_uniform {
                            cand_is_uniform
                        } else if win_is_uniform {
                            is_word_array_uniform_64(cand_w)
                        } else {
                            false
                        };
                        coherence_from_word_arrays_with_both_uniform_flags_64(
                            win_w,
                            win_is_uniform,
                            cand_w,
                            cand_uni_for_win,
                        )
                    }
                };
                clean_link * branchless_relu_f32(win_coh)
            } else {
                0.0
            };
            prior_term + link_term
        } else {
            0.0
        };
        let bayes_delta = branchless_flush_subnormal_f32(phase_scale * coh + eff_prior_link);

        if let Some(ref mut scores) = scores_slice_opt {
            if idx < scores.len() {
                let penalty = branchless_select_f32(is_coherent, 0.0, clean_penalty);
                scores[idx] = branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
            }
        }

        let take = is_coherent & ((!found) | (bayes_delta > best_score));
        if take {
            best_id = cand_id;
            best_idx = idx;
            best_score = bayes_delta;
            best_coh = coh;
            best_link_term = link_term;
            best_cand_is_uniform = cand_is_uniform;
            best_cand_eq_self = cand_eq_self;
        }
        found |= is_coherent;
        idx += 1;
    }

    if prune_non_macro & (!found) {
        fallback_scan_unpruned_candidates_dpsr_cold(
            &self_w,
            self_is_uniform,
            self_is_byte_uniform,
            self_b0,
            candidate_trails,
            primary_macro_anchor_id,
            secondary_macro_anchor_id,
            cur_linked_id,
            phase_scale,
            clean_eta,
            safe_thresh,
            &mut best_id,
            &mut best_idx,
            &mut best_score,
            &mut best_coh,
            &mut best_link_term,
            &mut best_cand_is_uniform,
            &mut found,
        );
    }

    #[allow(unused_mut)]
    let mut qdps_done = false;
    let trans_lp = if found {
        src_trail.linked_trail_id = best_id;
        if let Some(scores) = scores_slice_opt {
            if best_idx < scores.len() && best_link_term < clean_link {
                scores[best_idx] = branchless_flush_subnormal_f32(
                    scores[best_idx] + (clean_link - best_link_term),
                );
            }
        }
        let dst_trail = &mut candidate_trails[best_idx];
        if !best_cand_eq_self {
            let eff_s =
                branchless_select_u8(fine_interpolation_u7 > 128, 128, fine_interpolation_u7)
                    as u64;
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            if !best_cand_is_uniform {
                if eff_s > 0 && eff_s < 128 {
                    qdps_done = true;
                    self_w = unsafe {
                        let (bvc0, bvc1) = crate::traits::branchless::avx2_words64_to_vec64(
                            trail_phase_words_ref(dst_trail),
                        );
                        let (vr0, vr1) = crate::traits::branchless::avx2_slerp_vec64(
                            vs0, vs1, bvc0, bvc1, eff_s,
                        );
                        let bridged_w = crate::traits::branchless::avx2_vec64_to_words64(vr0, vr1);
                        *trail_phase_words_mut(src_trail) = bridged_w;
                        *trail_phase_words_mut(dst_trail) = bridged_w;
                        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
                        if step_u8 > 1 {
                            let (vf0, vf1) =
                                crate::traits::branchless::avx2_qdps_filter_vec64(vr0, vr1, step_u8);
                            crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                        } else {
                            bridged_w
                        }
                    };
                } else {
                    let dst_w = *trail_phase_words_ref(dst_trail);
                    if !swar_words_equal_64(&self_w, &dst_w) {
                        if eff_s >= 128 {
                            self_w = dst_w;
                            *trail_phase_words_mut(src_trail) = self_w;
                        } else {
                            *trail_phase_words_mut(dst_trail) = self_w;
                        }
                    }
                }
            } else {
                let dst_w = *trail_phase_words_ref(dst_trail);
                if !swar_words_equal_64(&self_w, &dst_w) {
                    if eff_s >= 128 {
                        self_w = dst_w;
                        *trail_phase_words_mut(src_trail) = self_w;
                    } else if eff_s > 0 {
                        let bias_restore =
                            (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let dst_uni = best_cand_is_uniform;
                        self_w = swar_slerp_words_with_uniform_flag_64(
                            &self_w,
                            self_is_uniform,
                            &dst_w,
                            dst_uni,
                            eff_s,
                            bias_restore,
                        )
                        .0;
                        *trail_phase_words_mut(src_trail) = self_w;
                        *trail_phase_words_mut(dst_trail) = self_w;
                    } else {
                        *trail_phase_words_mut(dst_trail) = self_w;
                    }
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            {
                let dst_w = *trail_phase_words_ref(dst_trail);
                if !swar_words_equal_64(&self_w, &dst_w) {
                    if eff_s >= 128 {
                        self_w = dst_w;
                        *trail_phase_words_mut(src_trail) = self_w;
                    } else if eff_s > 0 {
                        let bias_restore =
                            (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let dst_uni = best_cand_is_uniform;
                        self_w = swar_slerp_words_with_uniform_flag_64(
                            &self_w,
                            self_is_uniform,
                            &dst_w,
                            dst_uni,
                            eff_s,
                            bias_restore,
                        )
                        .0;
                        *trail_phase_words_mut(src_trail) = self_w;
                        *trail_phase_words_mut(dst_trail) = self_w;
                    } else {
                        *trail_phase_words_mut(dst_trail) = self_w;
                    }
                }
            }
        }

        dst_trail.prev_ptr = src_trail.current_ptr;
        dst_trail.step_count = src_trail.step_count.wrapping_add(1);
        let inherit_anchor = dst_trail.target_anchor_id != 0;
        src_trail.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            dst_trail.target_anchor_id,
            src_trail.target_anchor_id,
        );
        src_trail.cluster_id = branchless_select_u32(
            src_trail.target_anchor_id != 0,
            src_trail.target_anchor_id,
            src_trail.cluster_id,
        );
        dst_trail.target_anchor_id = src_trail.target_anchor_id;
        dst_trail.cluster_id = src_trail.cluster_id;

        fast_transition_log_prob_from_coh(best_coh)
    } else {
        self_w = [
            swar_q7_neg_sat_x8(self_w[0]),
            swar_q7_neg_sat_x8(self_w[1]),
            swar_q7_neg_sat_x8(self_w[2]),
            swar_q7_neg_sat_x8(self_w[3]),
            swar_q7_neg_sat_x8(self_w[4]),
            swar_q7_neg_sat_x8(self_w[5]),
            swar_q7_neg_sat_x8(self_w[6]),
            swar_q7_neg_sat_x8(self_w[7]),
        ];
        *trail_phase_words_mut(src_trail) = self_w;
        src_trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        let cur_ptr = gcso_tagged_ptr_t(src_trail.current_ptr);
        let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
        src_trail.current_ptr = cur_ptr
            .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
            .conjugate_phase()
            .raw();
        src_trail.cluster_id = ATTRACTOR_SLOT_REPULSION as u32;
        0.0_f32
    };

    filter_qdps_and_steer_from_words_64(
        &self_w,
        branchless_select_f32(qdps_done, 0.0, min_step_rad),
        max_rad,
        head_dim,
        num_heads,
        query_tensor,
        key_tensor,
    );

    let out_score = branchless_select_f32(found, best_score, 0.0);
    (best_id, out_score, trans_lp, found)
}

// ===================================================================
// 1. Blanket-Implemented Closed-Loop DPSR + Swarm / Unified-Field / Limina / Gateway Trait
// ===================================================================

/// Blanket-implemented closed-loop steering trait uniting [`PhaseSteering`] and [`PspmRouter`]
/// with [`StigmergicSwarmOps`], Single-Manifold Unified-Field resonance, Bayesian Pointer-Trail
/// selection & Attention score modulation, Orthogonal Procrustes trail feedback, Limina
/// inter-memory bridging, Online Toroidal $k$-Means / DP-Means Coarse-to-Fine routing, and
/// 2-Origin Composite Manifold Gateway execution.
///
/// Automatically implemented for all `T: PhaseSteering + PspmRouter + ?Sized` (including
/// [`DpsrEngine`], [`crate::traits::ActionHub`], [`crate::traits::HotPathObserver`],
/// [`crate::traits::ColdPathObserver`], and [`crate::traits::FullStackGcsoEngine`]).
pub trait DpsrClosedLoopSteering: PhaseSteering + PspmRouter {
    /// Executes closed-loop 8-Cell SWAR Micro-Swarm consensus on `trail` against `target_phases`
    /// via [`StigmergicSwarmOps::try_resolve_action_in_swarm_to_words`] and steers `query_tensor`
    /// (and optionally `key_tensor`) directly from the 8 returned `u64` SWAR registers via
    /// `steer_from_swar_words_core_safe` without reloading `trail.accumulated_phase_delta`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `target_phases.len() != MAX_ATTENTION_HEADS`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor bounds are invalid.
    #[inline(always)]
    fn resolve_and_apply_swarm_phase_steering_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        target_phases: &AlignedSlice16<'_, gcso_q7_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<SwarmResolutionReport> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;
        let tgt_arr = target_phases.as_array::<MAX_ATTENTION_HEADS>()?;
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.try_resolve_action_in_swarm_to_words(
            action_ptr,
            tgt_arr,
            0xFF,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Resolves `trail` in the 8-Cell SWAR Micro-Swarm directly on the **Single Unified Attractor
    /// Phase Manifold** (`anchor_phases`, `anchor_base_energies`) without requiring a 512-byte
    /// 8-slot phase bank or $8 \times 8$ coupling matrix.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor dimensions are invalid.
    #[inline(always)]
    fn resolve_unified_field_swarm_and_steer_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_paged_mask: Option<&gcso_paged_bitmask_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<SwarmResolutionReport> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let active_cell_mask = match active_paged_mask {
            Some(m) => {
                let mask8 = m.subswarm_activity_mask_8();
                branchless_select_u8(mask8 != 0, mask8, 0xFF)
            }
            None => 0xFF,
        };
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.resolve_unified_field_in_swarm_to_words(
            action_ptr,
            anchor_phases,
            anchor_base_energies,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Resolves `trail` in the 8-Cell Micro-Swarm against the optional 8-Lane Working Phase Bank
    /// (`slot_phase_bank`, `>= 512` bytes), recruits coupled neighbor lanes via
    /// [`StigmergicSwarmOps::resolve_slot_bank_in_swarm_to_words`] on partial quorum,
    /// and steers `query_tensor` (and optionally `key_tensor`) directly from the register-resident
    /// `swar_words` with geodesic RIPA soft-clamping.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`]
    /// if `slot_phase_bank.len() < 512`, `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor bounds fail.
    #[inline(always)]
    fn resolve_slot_bank_swarm_and_steer_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_paged_mask: Option<&gcso_paged_bitmask_t>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<SwarmResolutionReport> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;
        let bank_2d =
            slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;

        let active_cell_mask = match active_paged_mask {
            Some(m) => {
                let mask8 = m.subswarm_activity_mask_8();
                branchless_select_u8(mask8 != 0, mask8, 0xFF)
            }
            None => 0xFF,
        };
        let qdps_min_step_q7 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let action_ptr = trail.current_tagged_ptr();

        let (report, swar_words) = trail.resolve_slot_bank_in_swarm_to_words(
            action_ptr,
            bank_2d,
            slot_weights,
            coupling_matrix_8x8,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        finish_constant_time_swarm_qdps_and_steer(
            report.evolved_ptr,
            report.quorum_mask,
            &swar_words,
            qdps_min_step_q7,
            unresolved_damping_u7,
            head_dim,
            num_heads,
            max_rad,
            trail,
            query_tensor,
            key_tensor,
        );
        Ok(report)
    }

    /// Directly modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`)
    /// in-place in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per
    /// candidate directly from `candidate_trails: &[gcso_pointer_trail_t]` using
    /// [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`].
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_id, coherent_candidate_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `candidate_trails` is
    /// empty, `head_dim()` is zero or odd, or any gain/threshold parameter is non-finite.
    #[inline(always)]
    fn modulate_candidate_trail_attention_scores_zero_copy(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        src_trail: &gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
    ) -> GcsoResult<(f32, u32, u32)> {
        let head_dim = self.head_dim();
        let count = attention_scores.len().min(candidate_trails.len());
        if count == 0
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(repulsion_penalty)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        Ok(src_trail.modulate_candidate_trail_scores_in_place(
            attention_scores.as_mut_slice(),
            candidate_trails,
            head_dim,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            contradiction_threshold,
            repulsion_penalty,
        ))
    }

    /// Realizes the single-pass closed-loop synthesis of **Hypothesis 1** (Persona as Joint
    /// Probability Chain Rule $P(M_1, \dots, M_T) = \prod_t P(M_t \mid M_{<t})$ $\times$ Attention
    /// phase interference) and **Hypothesis 2** (Intelligence as Stigmergic Pointer-Trail
    /// manipulation in `ActionHub`) directly on any `T: PhaseSteering + PspmRouter` in zero-copy,
    /// zero-allocation ($C_{\mathrm{alloc}} = 0$) mode.
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_coherence, transition_log_prob, found_coherent_transition)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn select_rewire_and_steer_bayesian_pointer_trail_safe(
        &self,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(u32, f32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let cand_len = candidate_trails.len();
        let safe_dim = branchless_select_u32(head_dim > 0, head_dim as u32, 128);
        let rsqrt_d = if safe_dim <= 256 {
            branchless_popcount_rsqrt_u32(safe_dim)
        } else {
            branchless_rsqrt_f32(safe_dim as f32)
        };
        let phase_scale =
            branchless_flush_subnormal_f32(branchless_relu_f32(phase_interference_gain) * rsqrt_d);
        let clean_eta = branchless_flush_subnormal_f32(branchless_relu_f32(stigmergic_prior_gain));
        let clean_link = branchless_flush_subnormal_f32(branchless_relu_f32(pointer_link_bonus));
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let clean_penalty = branchless_flush_subnormal_f32(branchless_relu_f32(repulsion_penalty));

        let mut self_w = *trail_phase_words_ref(src_trail);
        let self_is_uniform = is_word_array_uniform_64(&self_w);
        let self_w0 = self_w[0];
        let self_b0 = (self_w0 & 0xFF) as u8;
        let self_is_byte_uniform = self_is_uniform & (self_w0 == self_w0.rotate_right(8));
        let cur_linked_id = src_trail.linked_trail_id;
        let mut scores_slice_opt = attention_scores.map(AlignedSliceMut32::as_mut_slice);

        let mut best_id = cur_linked_id;
        let mut best_idx = 0usize;
        let mut best_score = -1.0e30_f32;
        let mut best_coh = 0.0_f32;
        let mut best_cand_is_uniform = false;
        let mut best_cand_eq_self = false;
        let mut found = false;
        let mut coh_sum = 0.0_f32;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (vs0, vs1) = unsafe {
            crate::traits::branchless::avx2_words64_to_vec64_with_uniform(&self_w, self_is_uniform)
        };

        let mut idx = 0usize;
        while idx < cand_len {
            let cand = &candidate_trails[idx];
            let cand_not_repulsed = (cand.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
            if (!cand_not_repulsed) & scores_slice_opt.is_none() {
                idx += 1;
                continue;
            }
            let cand_id = idx as u32;
            let cand_w = trail_phase_words_ref(cand);
            let cw0 = cand_w[0];
            let cand_eq_self = (cw0 == self_w0) && swar_words_equal_64(&self_w, cand_w);
            let cand_is_uniform = if cand_eq_self {
                self_is_uniform
            } else {
                self_is_uniform && is_word_array_uniform_64(cand_w)
            };
            let coh = if cand_eq_self {
                1.0
            } else if cand_is_uniform {
                if self_is_byte_uniform && (cw0 == cw0.rotate_right(8)) {
                    Q7_FULL_COS_REF[self_b0.wrapping_sub(cw0 as u8) as usize]
                } else {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w, true, cand_w, true,
                    )
                }
            } else {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    unsafe {
                        use core::arch::x86_64::_mm256_sub_epi8;
                        let (vc0, vc1) = crate::traits::branchless::avx2_words64_to_vec64(cand_w);
                        crate::traits::branchless::avx2_coherence_from_diff_vec64(
                            _mm256_sub_epi8(vs0, vc0),
                            _mm256_sub_epi8(vs1, vc1),
                        )
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    coherence_from_word_arrays_with_both_uniform_flags_64(
                        &self_w,
                        self_is_uniform,
                        cand_w,
                        false,
                    )
                }
            };
            let is_coherent = (coh >= -safe_thresh) & cand_not_repulsed;
            let eff_prior_link = if is_coherent {
                let prior_term = clean_eta * fast_ln_1p_bounded_f32(cand.stigmergic_density);
                let is_linked = (cur_linked_id != 0) & (cur_linked_id == cand_id);
                prior_term + branchless_select_f32(is_linked, clean_link, 0.0)
            } else {
                0.0
            };
            let bayes_delta = branchless_flush_subnormal_f32(phase_scale * coh + eff_prior_link);

            if let Some(ref mut scores) = scores_slice_opt {
                if idx < scores.len() {
                    let penalty = branchless_select_f32(is_coherent, 0.0, clean_penalty);
                    scores[idx] =
                        branchless_flush_subnormal_f32(scores[idx] + bayes_delta - penalty);
                }
            }

            let take = is_coherent & ((!found) | (bayes_delta > best_score));
            if take {
                best_id = cand_id;
                best_idx = idx;
                best_score = bayes_delta;
                best_coh = coh;
                best_cand_is_uniform = cand_is_uniform;
                best_cand_eq_self = cand_eq_self;
            }
            found |= is_coherent;
            coh_sum += coh;
            idx += 1;
        }

        #[allow(unused_mut)]
        let mut qdps_done = false;
        let (coh, log_prob) = if found {
            src_trail.linked_trail_id = best_id;
            if let Some(scores) = scores_slice_opt {
                if best_idx < scores.len() && cur_linked_id != best_id {
                    scores[best_idx] =
                        branchless_flush_subnormal_f32(scores[best_idx] + clean_link);
                }
            }
            let dst_trail = &mut candidate_trails[best_idx];
            if !best_cand_eq_self {
                let eff_s =
                    branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if !best_cand_is_uniform {
                    if eff_s > 0 && eff_s < 128 {
                        qdps_done = true;
                        self_w = unsafe {
                            let (bvc0, bvc1) = crate::traits::branchless::avx2_words64_to_vec64(
                                trail_phase_words_ref(dst_trail),
                            );
                            let (vr0, vr1) = crate::traits::branchless::avx2_slerp_vec64(
                                vs0, vs1, bvc0, bvc1, eff_s,
                            );
                            let bridged_w =
                                crate::traits::branchless::avx2_vec64_to_words64(vr0, vr1);
                            *trail_phase_words_mut(src_trail) = bridged_w;
                            *trail_phase_words_mut(dst_trail) = bridged_w;
                            let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
                            if step_u8 > 1 {
                                let (vf0, vf1) = crate::traits::branchless::avx2_qdps_filter_vec64(
                                    vr0, vr1, step_u8,
                                );
                                crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                            } else {
                                bridged_w
                            }
                        };
                    } else {
                        let dst_w = *trail_phase_words_ref(dst_trail);
                        if !swar_words_equal_64(&self_w, &dst_w) {
                            if eff_s >= 128 {
                                self_w = dst_w;
                                *trail_phase_words_mut(src_trail) = self_w;
                            } else {
                                *trail_phase_words_mut(dst_trail) = self_w;
                            }
                        }
                    }
                } else {
                    let dst_w = *trail_phase_words_ref(dst_trail);
                    if !swar_words_equal_64(&self_w, &dst_w) {
                        if eff_s >= 128 {
                            self_w = dst_w;
                            *trail_phase_words_mut(src_trail) = self_w;
                        } else if eff_s > 0 {
                            let bias_restore =
                                (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let dst_uni = best_cand_is_uniform;
                            self_w = swar_slerp_words_with_uniform_flag_64(
                                &self_w,
                                self_is_uniform,
                                &dst_w,
                                dst_uni,
                                eff_s,
                                bias_restore,
                            )
                            .0;
                            *trail_phase_words_mut(src_trail) = self_w;
                            *trail_phase_words_mut(dst_trail) = self_w;
                        } else {
                            *trail_phase_words_mut(dst_trail) = self_w;
                        }
                    }
                }
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                {
                    let dst_w = *trail_phase_words_ref(dst_trail);
                    if !swar_words_equal_64(&self_w, &dst_w) {
                        if eff_s >= 128 {
                            self_w = dst_w;
                            *trail_phase_words_mut(src_trail) = self_w;
                        } else if eff_s > 0 {
                            let bias_restore =
                                (128u64.wrapping_sub(eff_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                            let dst_uni = best_cand_is_uniform;
                            self_w = swar_slerp_words_with_uniform_flag_64(
                                &self_w,
                                self_is_uniform,
                                &dst_w,
                                dst_uni,
                                eff_s,
                                bias_restore,
                            )
                            .0;
                            *trail_phase_words_mut(src_trail) = self_w;
                            *trail_phase_words_mut(dst_trail) = self_w;
                        } else {
                            *trail_phase_words_mut(dst_trail) = self_w;
                        }
                    }
                }
            }

            dst_trail.prev_ptr = src_trail.current_ptr;
            dst_trail.step_count = src_trail.step_count.wrapping_add(1);
            let inherit_anchor = dst_trail.target_anchor_id != 0;
            src_trail.target_anchor_id = branchless_select_u32(
                inherit_anchor,
                dst_trail.target_anchor_id,
                src_trail.target_anchor_id,
            );
            src_trail.cluster_id = branchless_select_u32(
                src_trail.target_anchor_id != 0,
                src_trail.target_anchor_id,
                src_trail.cluster_id,
            );
            dst_trail.target_anchor_id = src_trail.target_anchor_id;
            dst_trail.cluster_id = src_trail.cluster_id;

            let lp = fast_transition_log_prob_from_coh(best_coh);
            (best_coh, lp)
        } else {
            self_w = [
                swar_q7_neg_sat_x8(self_w[0]),
                swar_q7_neg_sat_x8(self_w[1]),
                swar_q7_neg_sat_x8(self_w[2]),
                swar_q7_neg_sat_x8(self_w[3]),
                swar_q7_neg_sat_x8(self_w[4]),
                swar_q7_neg_sat_x8(self_w[5]),
                swar_q7_neg_sat_x8(self_w[6]),
                swar_q7_neg_sat_x8(self_w[7]),
            ];
            *trail_phase_words_mut(src_trail) = self_w;
            src_trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
            let cur_ptr = gcso_tagged_ptr_t(src_trail.current_ptr);
            let rep_mode = cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE;
            src_trail.current_ptr = cur_ptr
                .with_action_and_slot(ATTRACTOR_SLOT_REPULSION, rep_mode)
                .conjugate_phase()
                .raw();
            src_trail.cluster_id = ATTRACTOR_SLOT_REPULSION as u32;
            let mean_coh = branchless_clamp_f32(coh_sum * fast_inv_len_f32(cand_len), -1.0, 1.0);
            (mean_coh, 0.0)
        };

        filter_qdps_and_steer_from_words_64(
            &self_w,
            branchless_select_f32(qdps_done, 0.0, min_step_rad),
            max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );

        let out_score = branchless_select_f32(found, best_score, 0.0);
        Ok((best_id, out_score, coh, log_prob, found))
    }

    /// Combines **Order-Invariant Single-Manifold Unified-Field Resonance**, **RIPA Q/K Phase
    /// Steering**, optional **Register-Resident $SO(2)$-Fused PSPM Sub-Head Pseudo-MoE**
    /// (`dispatch_fused_ripa_and_pspm_words_core`, eliminating the 64-byte `phase_buf`
    /// stack spill), and optional **Multiplicative Inverse-Temperature Logit Scaling**
    /// ($\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$)
    /// without requiring a 512-byte 8-slot phase bank.
    ///
    /// Returns `(final_pull_force, mean_coherence, applied_inv_temp_scale, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor/PSPM parameters are invalid.
    #[inline(always)]
    fn steer_unified_field_pspm_and_inv_temperature_safe(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        pspm_config_and_explore_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        logits_and_inv_temp_gain: Option<(&mut AlignedSliceMut32<'_, f32>, f32)>,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some((_, strength)) = repulsion_phases_and_strength {
            if !branchless_is_finite_f32(strength) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let init_pull = trail.attractor_pull_force;
        let (final_pull, mean_coh, active_cnt, raw_w) = trail
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );

        let need_dom_w =
            logits_and_inv_temp_gain.is_some() || pspm_config_and_explore_pulse.is_some();
        let raw_eq_dom = (!need_dom_w)
            | ((active_cnt <= 1)
                & ((anchor_phases.len() <= 1) | (init_pull <= 1.0e-12))
                & repulsion_phases_and_strength.is_none());
        let dom_w = if raw_eq_dom || anchor_phases.is_empty() {
            raw_w
        } else {
            let dom_idx_usize = (trail.target_anchor_id as usize).min(anchor_phases.len() - 1);
            load_q7_words_64(&anchor_phases[dom_idx_usize])
        };
        let dom_eq_raw =
            raw_eq_dom || anchor_phases.len() <= 1 || swar_words_equal_64(&raw_w, &dom_w);

        let mut inv_temp = 1.0_f32;
        if let Some((logits, temp_gain)) = logits_and_inv_temp_gain {
            if logits.is_empty() || !branchless_is_finite_f32(temp_gain) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            let coh_dom = if anchor_phases.len() <= 1 {
                mean_coh
            } else if dom_eq_raw {
                1.0_f32
            } else {
                let raw_uni = is_word_array_uniform_64(&raw_w);
                coherence_from_word_arrays_with_both_uniform_flags_64(
                    &raw_w,
                    raw_uni,
                    &dom_w,
                    raw_uni && is_word_array_uniform_64(&dom_w),
                )
            };
            inv_temp = branchless_phase_coherence_inv_temp_scale(coh_dom, temp_gain);
            scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        }

        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let filt_w = filter_qdps_words_64(&raw_w, step_u8);
        let filt_nonzero = swar_words_nonzero_64(&filt_w);

        if let Some((pspm_cfg, explore_pulse_q7)) = pspm_config_and_explore_pulse {
            let total_pspm_heads = (pspm_cfg.num_fact_heads as usize)
                .checked_add(pspm_cfg.num_logic_heads as usize)
                .and_then(|v| v.checked_add(pspm_cfg.num_explore_heads as usize))
                .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
            if total_pspm_heads > num_heads {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }

            let filt_eq_dom = if step_u8 <= 1 {
                dom_eq_raw
            } else {
                dom_eq_raw || swar_words_equal_64(&filt_w, &dom_w)
            };
            let is_default_ripa_pspm = filt_eq_dom
                & (explore_pulse_q7 == 0)
                & (total_pspm_heads == num_heads)
                & (pspm_cfg.fact_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (pspm_cfg.logic_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (pspm_cfg.explore_phase_gain.to_bits() == 1.0_f32.to_bits())
                & (max_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                & (num_heads > 0)
                & (num_heads <= MAX_ATTENTION_HEADS);
            if is_default_ripa_pspm {
                let rot_dim = ((head_dim >> 2) & !1).max(2).min(head_dim);
                if let Some(k) = key_tensor.as_deref_mut() {
                    if filt_nonzero {
                        crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            query_tensor.as_mut_slice(),
                            k.as_mut_slice(),
                            &filt_w,
                            num_heads,
                            head_dim,
                            rot_dim,
                        );
                    }
                } else if filt_nonzero {
                    crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                        query_tensor.as_mut_slice(),
                        &filt_w,
                        num_heads,
                        head_dim,
                        rot_dim,
                        true,
                    );
                }
            } else {
                let (fact_w_ref, logic_w_ref, explore_w_ref) = if filt_eq_dom {
                    (&filt_w, &filt_w, &filt_w)
                } else {
                    (&dom_w, &filt_w, &filt_w)
                };
                dispatch_fused_ripa_and_pspm_words_core(
                    query_tensor,
                    &filt_w,
                    max_rad,
                    pspm_cfg,
                    fact_w_ref,
                    logic_w_ref,
                    explore_w_ref,
                    max_rad,
                    explore_pulse_q7,
                    head_dim,
                    true,
                )?;
                if let Some(k) = key_tensor.as_deref_mut() {
                    if filt_nonzero {
                        steer_from_swar_words_core_safe(
                            &filt_w, 0xFF, 128, head_dim, num_heads, false, max_rad, k, None,
                        );
                    }
                }
            }
            if total_pspm_heads < num_heads && filt_nonzero {
                let tail_elems = (num_heads - total_pspm_heads) * head_dim;
                let q_tail = &mut query_tensor.as_mut_slice()
                    [total_pspm_heads * head_dim..total_pspm_heads * head_dim + tail_elems];
                let mut tail_words = [0u64; NUM_SWARM_MICRO_CELLS];
                let mut h = total_pspm_heads;
                while h < num_heads {
                    let th = h - total_pspm_heads;
                    let byte_val = (filt_w[h >> 3] >> ((h & 7) << 3)) & 0xFF;
                    tail_words[th >> 3] |= byte_val << ((th & 7) << 3);
                    h += 1;
                }
                if swar_words_nonzero_64(&tail_words) {
                    // SAFETY: `total_pspm_heads * head_dim` is a multiple of `head_dim * 4` bytes;
                    // for residual heads we rotate each upper `d_head / 4` slice directly.
                    let mut tail_aligned = unsafe { AlignedSliceMut32::new_unchecked(q_tail) };
                    steer_from_swar_words_core_safe(
                        &tail_words,
                        0xFF,
                        128,
                        head_dim,
                        num_heads - total_pspm_heads,
                        false,
                        max_rad,
                        &mut tail_aligned,
                        None,
                    );
                }
            }
        } else if filt_nonzero {
            steer_from_swar_words_core_safe(
                &filt_w,
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

        Ok((final_pull, mean_coh, inv_temp, trail.target_anchor_id))
    }

    /// Computes the closed-form Orthogonal Procrustes $SO(2)$ phase shift between `query_tensor`
    /// and `target_key_tensor` via [`PhaseSteering::compute_procrustes_phase_delta`], applies
    /// constant-time 8-lane SWAR gain scaling (`swar_q7_scale_u7_x8`) and QDPS step filtering
    /// (`swar_q7_qdps_filter_x8`), accumulates circularly into `trail.accumulated_phase_delta`
    /// (`swar_q7_circular_add_x8`), and steers `query_tensor` in-place via
    /// `steer_from_swar_words_core_safe`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `min_step_rad < 0.0`, `max_rad <= 0.0`, or tensor lengths are insufficient.
    #[inline(always)]
    fn steer_trail_with_procrustes_feedback(
        &self,
        trail: &mut gcso_pointer_trail_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        target_key_tensor: &AlignedSlice32<'_, f32>,
        feedback_gain_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        let step_bits = min_step_rad.to_bits();
        let step_abs_bits = step_bits & 0x7FFF_FFFF;
        if ((step_bits >> 31) != 0 && step_abs_bits != 0)
            || (step_bits & 0x7F80_0000) == 0x7F80_0000
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, None)?;
        let expected_len = head_dim * num_heads;
        if target_key_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let clean_gain = branchless_select_u8(feedback_gain_u7 > 128, 128, feedback_gain_u7);
        if clean_gain == 0 {
            let nw = *trail_phase_words_ref(trail);
            if swar_words_nonzero_64(&nw) {
                steer_from_swar_words_core_safe(
                    &nw,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    true,
                    max_rad,
                    query_tensor,
                    None,
                );
            }
            return Ok(AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]));
        }

        let gain_u64 = u64::from(clean_gain);
        let gain_bias = (128u64.wrapping_sub(gain_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let step_u8 = if step_abs_bits <= Q7_PHASE_SCALE_RAD.to_bits() {
            (step_abs_bits != 0) as u8
        } else {
            branchless_rad_to_qdps_step_u8(min_step_rad)
        };
        let active_cells = num_heads
            .div_ceil(HEADS_PER_SWARM_CELL)
            .min(NUM_SWARM_MICRO_CELLS);
        let is_unit_gain = clean_gain == 128;
        let is_bypass_step = step_u8 <= 1;

        // Two-stage DSB-locked execution:
        // Stage 1 (Read-Only SIMD Procrustes Extraction): Runs `procrustes_q7_words_64_simd` across
        // all active 8-head cells while keeping L1D cache ports 100% in read mode and preserving
        // AVX2 `atan2` polynomial constants in YMM registers without store-turnaround stalls.
        // Stage 2 (Register-Resident SWAR Trail Accum + Coalesced Geodesic RIPA Steering): Updates
        // `trail.accumulated_phase_delta` in registers and invokes `steer_from_swar_words_core_safe`
        // while the upper `d_head / 4` channels of `query_tensor` remain 100% hot in L1D cache.
        let mut raw_words = procrustes_q7_words_64_simd(
            query_tensor.as_slice(),
            target_key_tensor.as_slice(),
            head_dim,
            num_heads,
        );
        let mut any_nonzero = 0u64;
        {
            let tw = trail_phase_words_mut(trail);
            if is_unit_gain & is_bypass_step {
                let mut c = 0usize;
                while c < active_cells {
                    let filt_w = raw_words[c];
                    let nw = swar_q7_circular_add_x8(tw[c], filt_w);
                    tw[c] = nw;
                    any_nonzero |= nw;
                    c += 1;
                }
            } else if is_bypass_step {
                let mut c = 0usize;
                while c < active_cells {
                    let filt_w =
                        swar_q7_scale_le128_with_bias_x8(raw_words[c], gain_u64, gain_bias);
                    raw_words[c] = filt_w;
                    let nw = swar_q7_circular_add_x8(tw[c], filt_w);
                    tw[c] = nw;
                    any_nonzero |= nw;
                    c += 1;
                }
            } else if is_unit_gain {
                let mut c = 0usize;
                while c < active_cells {
                    let filt_w = swar_q7_qdps_filter_x8(raw_words[c], step_u8);
                    raw_words[c] = filt_w;
                    let nw = swar_q7_circular_add_x8(tw[c], filt_w);
                    tw[c] = nw;
                    any_nonzero |= nw;
                    c += 1;
                }
            } else {
                let mut c = 0usize;
                while c < active_cells {
                    let scaled_w =
                        swar_q7_scale_le128_with_bias_x8(raw_words[c], gain_u64, gain_bias);
                    let filt_w = swar_q7_qdps_filter_x8(scaled_w, step_u8);
                    raw_words[c] = filt_w;
                    let nw = swar_q7_circular_add_x8(tw[c], filt_w);
                    tw[c] = nw;
                    any_nonzero |= nw;
                    c += 1;
                }
            }
        }
        if any_nonzero != 0 {
            steer_from_swar_words_core_safe(
                trail_phase_words_ref(trail),
                0xFF,
                128,
                head_dim,
                num_heads,
                true,
                max_rad,
                query_tensor,
                None,
            );
        }
        Ok(AlignedBuffer16::new(words_to_q7_array_64(&raw_words)))
    }

    /// Bridges the inter-memory **Limina** transition from `prev_memory_trail` ($M_{t-1}$) to
    /// `curr_trail` ($M_t$) via [`StigmergicTrailOps::bridge_liminal_hallucinated_trail`],
    /// filters `curr_trail.accumulated_phase_delta` in registers via `filter_qdps_and_steer_from_phases_64`,
    /// steers `query_tensor` (and optional `key_tensor`) with RIPA soft-clamping, and optionally
    /// modulates candidate memory pre-Softmax Attention scores (`attention_scores`) by
    /// Attention $\times$ DPSR phase interference.
    ///
    /// Returns `(transition_coherence, transition_log_prob, contradiction_repulsed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `contradiction_threshold`, `min_step_rad`, or
    /// `max_rad` is invalid, or if tensor slice dimensions are insufficient.
    #[inline(always)]
    fn bridge_liminal_trail_and_steer_qk_safe(
        &self,
        prev_memory_trail: &gcso_pointer_trail_t,
        curr_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        min_step_rad: f32,
        max_rad: f32,
        attention_interference_gain: f32,
        memory_anchor_phases: Option<&[[gcso_q7_t; MAX_ATTENTION_HEADS]]>,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, bool)> {
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let t_s = branchless_select_u8(interpolation_u7 > 128, 128, interpolation_u7) as u64;

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let (coh, repulsed, out_w, log_prob, vo0, vo1) = unsafe {
            use core::arch::x86_64::{
                _mm256_max_epi8, _mm256_or_si256, _mm256_set1_epi8, _mm256_setzero_si256,
                _mm256_sub_epi8, _mm256_testz_si256,
            };
            let p_words = *trail_phase_words_ref(prev_memory_trail);
            let s_words = *trail_phase_words_ref(curr_trail);
            let (vp0, vp1) = crate::traits::branchless::avx2_words64_to_vec64(&p_words);
            let (vs0, vs1) = crate::traits::branchless::avx2_words64_to_vec64(&s_words);
            let vd0 = _mm256_sub_epi8(vs0, vp0);
            let vd1 = _mm256_sub_epi8(vs1, vp1);
            let vd_or = _mm256_or_si256(vd0, vd1);
            if _mm256_testz_si256(vd_or, vd_or) != 0 {
                (1.0_f32, false, s_words, 0.0_f32, vs0, vs1)
            } else {
                let c = crate::traits::branchless::avx2_coherence_from_diff_vec64(vd0, vd1);
                let rep = c < -safe_thresh;
                let (v_out0, v_out1, store_out) = if rep {
                    let zero = _mm256_setzero_si256();
                    let min_m127 = _mm256_set1_epi8(-127);
                    let r0 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs0, min_m127));
                    let r1 = _mm256_sub_epi8(zero, _mm256_max_epi8(vs1, min_m127));
                    (r0, r1, true)
                } else if t_s >= 128 {
                    (vs0, vs1, false)
                } else if t_s == 0 {
                    (vp0, vp1, true)
                } else {
                    let (r0, r1) = crate::traits::branchless::avx2_slerp_from_diff_vec64(
                        vp0, vp1, vd0, vd1, t_s,
                    );
                    (r0, r1, true)
                };
                let ow = crate::traits::branchless::avx2_vec64_to_words64(v_out0, v_out1);
                if store_out {
                    *trail_phase_words_mut(curr_trail) = ow;
                }
                let lp = fast_transition_log_prob_from_coh(c);
                (c, rep, ow, lp, v_out0, v_out1)
            }
        };

        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let (coh, repulsed, out_w, log_prob) = {
            use crate::traits::branchless::{
                branchless_cos_q7, coherence_from_diff_words_64, swar_q7_circular_diff_x8,
            };
            let p = *trail_phase_words_ref(prev_memory_trail);
            let s = *trail_phase_words_ref(curr_trail);
            let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            if (s[0] == p[0]) && swar_words_equal_64(&s, &p) {
                (1.0_f32, false, s, 0.0_f32)
            } else {
                let d0 = swar_q7_circular_diff_x8(s[0], p[0]);
                let d1 = swar_q7_circular_diff_x8(s[1], p[1]);
                let d2 = swar_q7_circular_diff_x8(s[2], p[2]);
                let d3 = swar_q7_circular_diff_x8(s[3], p[3]);
                let d4 = swar_q7_circular_diff_x8(s[4], p[4]);
                let d5 = swar_q7_circular_diff_x8(s[5], p[5]);
                let d6 = swar_q7_circular_diff_x8(s[6], p[6]);
                let d7 = swar_q7_circular_diff_x8(s[7], p[7]);
                let diff_is_uniform = (d0 == d1)
                    && ((((d0 ^ d2) | (d0 ^ d3)) | ((d0 ^ d4) | (d0 ^ d5)))
                        | ((d0 ^ d6) | (d0 ^ d7)))
                        == 0;
                let c = if diff_is_uniform && (d0 == d0.rotate_right(8)) {
                    branchless_cos_q7(d0 as gcso_q7_t)
                } else {
                    coherence_from_diff_words_64(d0, d1, d2, d3, d4, d5, d6, d7)
                };
                let rep = c < -safe_thresh;
                let ow = if rep {
                    [
                        swar_q7_neg_sat_x8(s[0]),
                        swar_q7_neg_sat_x8(s[1]),
                        swar_q7_neg_sat_x8(s[2]),
                        swar_q7_neg_sat_x8(s[3]),
                        swar_q7_neg_sat_x8(s[4]),
                        swar_q7_neg_sat_x8(s[5]),
                        swar_q7_neg_sat_x8(s[6]),
                        swar_q7_neg_sat_x8(s[7]),
                    ]
                } else if t_s >= 128 {
                    s
                } else if t_s == 0 {
                    p
                } else if diff_is_uniform {
                    let st0 = swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore);
                    [
                        swar_q7_circular_add_x8(p[0], st0),
                        swar_q7_circular_add_x8(p[1], st0),
                        swar_q7_circular_add_x8(p[2], st0),
                        swar_q7_circular_add_x8(p[3], st0),
                        swar_q7_circular_add_x8(p[4], st0),
                        swar_q7_circular_add_x8(p[5], st0),
                        swar_q7_circular_add_x8(p[6], st0),
                        swar_q7_circular_add_x8(p[7], st0),
                    ]
                } else {
                    [
                        swar_q7_circular_add_x8(
                            p[0],
                            swar_q7_scale_le128_with_bias_x8(d0, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[1],
                            swar_q7_scale_le128_with_bias_x8(d1, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[2],
                            swar_q7_scale_le128_with_bias_x8(d2, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[3],
                            swar_q7_scale_le128_with_bias_x8(d3, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[4],
                            swar_q7_scale_le128_with_bias_x8(d4, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[5],
                            swar_q7_scale_le128_with_bias_x8(d5, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[6],
                            swar_q7_scale_le128_with_bias_x8(d6, t_s, bias_restore),
                        ),
                        swar_q7_circular_add_x8(
                            p[7],
                            swar_q7_scale_le128_with_bias_x8(d7, t_s, bias_restore),
                        ),
                    ]
                };
                let lp = fast_transition_log_prob_from_coh(c);
                *trail_phase_words_mut(curr_trail) = ow;
                (c, rep, ow, lp)
            }
        };

        curr_trail.prev_ptr = prev_memory_trail.current_ptr;
        curr_trail.step_count = prev_memory_trail.step_count.wrapping_add(1);
        let inherit_anchor = (curr_trail.target_anchor_id == 0) & (!repulsed);
        curr_trail.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            prev_memory_trail.target_anchor_id,
            curr_trail.target_anchor_id,
        );
        curr_trail.cluster_id = branchless_select_u32(
            inherit_anchor & (prev_memory_trail.target_anchor_id != 0),
            prev_memory_trail.target_anchor_id,
            curr_trail.cluster_id,
        );
        let cur_ptr = gcso_tagged_ptr_t(curr_trail.current_ptr);
        let rep_ptr = cur_ptr
            .with_action_and_slot(
                ATTRACTOR_SLOT_REPULSION,
                cur_ptr.action_mode_flags() | ACTION_PRESET_REPULSION_ESCAPE,
            )
            .conjugate_phase();
        curr_trail.current_ptr = branchless_select_u64(repulsed, rep_ptr.raw(), cur_ptr.raw());

        let cur_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.stigmergic_density));
        let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(coh));
        let neg_rho = cur_rho * 0.5;
        curr_trail.stigmergic_density = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(repulsed, neg_rho, pos_rho)),
            MAX_STIGMERGIC_DENSITY,
        );

        let prev_pull = branchless_flush_subnormal_f32(branchless_relu_f32(
            prev_memory_trail.attractor_pull_force,
        ));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(curr_trail.attractor_pull_force));
        let bridged_pull = cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
        curr_trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_select_f32(
                repulsed,
                cur_pull * 0.5,
                bridged_pull,
            )),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let is_resonant = (!repulsed) & (coh > 0.25);
        curr_trail.flags |= branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(
                repulsed,
                TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
                0,
            );

        if let (Some(scores), Some(anchors)) = (attention_scores, memory_anchor_phases) {
            if !anchors.is_empty() && !scores.is_empty() {
                if !branchless_is_finite_f32(attention_interference_gain) {
                    return Err(GCSO_ERROR_INVALID_ARGUMENT);
                }
                let count = scores.len().min(anchors.len());
                let safe_dim = branchless_select_u32(head_dim > 0, head_dim as u32, 128);
                let rsqrt_d = if safe_dim <= 256 {
                    branchless_popcount_rsqrt_u32(safe_dim)
                } else {
                    branchless_rsqrt_f32(safe_dim as f32)
                };
                let eff_scale = branchless_flush_subnormal_f32(
                    branchless_relu_f32(attention_interference_gain) * rsqrt_d,
                );
                let score_slice = &mut scores.as_mut_slice()[..count];
                #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                let out_is_uniform = is_word_array_uniform_64(&out_w);
                let mut i = 0usize;
                while i < count {
                    let c_j = {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            // SAFETY: `anchors[i]` is 64 bytes (`2 * __m256i`) and `vo0, vo1` hold `out_w`.
                            unsafe {
                                use core::arch::x86_64::{
                                    __m256i, _mm256_loadu_si256, _mm256_sub_epi8,
                                };
                                let ap = anchors[i].as_ptr().cast::<__m256i>();
                                let va0 = _mm256_loadu_si256(ap);
                                let va1 = _mm256_loadu_si256(ap.add(1));
                                crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vo0, va0),
                                    _mm256_sub_epi8(vo1, va1),
                                )
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            let anc_w = load_q7_words_64(&anchors[i]);
                            coherence_from_word_arrays_with_both_uniform_flags_64(
                                &out_w,
                                out_is_uniform,
                                &anc_w,
                                out_is_uniform && is_word_array_uniform_64(&anc_w),
                            )
                        }
                    };
                    score_slice[i] =
                        branchless_flush_subnormal_f32(score_slice[i] + eff_scale * c_j);
                    i += 1;
                }
            }
        }

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
            let steer_w = if step_u8 > 1 {
                // SAFETY: `target_feature = "avx2"` is enabled; `vo0, vo1` are already in YMM registers.
                unsafe {
                    let (vf0, vf1) =
                        crate::traits::branchless::avx2_qdps_filter_vec64(vo0, vo1, step_u8);
                    crate::traits::branchless::avx2_vec64_to_words64(vf0, vf1)
                }
            } else {
                out_w
            };
            filter_qdps_and_steer_from_words_64(
                &steer_w,
                0.0,
                max_rad,
                head_dim,
                num_heads,
                query_tensor,
                key_tensor,
            );
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        filter_qdps_and_steer_from_words_64(
            &out_w,
            min_step_rad,
            max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok((coh, log_prob, repulsed))
    }

    /// Synthesizes any [`ManifoldGatewaySignal`] (such as a 2/3/4-tuple of
    /// `(SystemConstraintSignal, ObjectiveFactSignal, SubjectivePersonaSignal, ContradictionGuardSignal)`,
    /// a [`GatewayFrame`], or a [`CrystallizedAttractorRecord`]), folds its resonant 64-head phase
    /// and 48-bit ZIMMS tagged pointer into `trail` in 8 register-resident `u64` SWAR words,
    /// executes $SO(2)$ group-closure fused RIPA + PSPM sub-head routing on `query_tensor` when
    /// `ACTION_MODE_PSPM_ROUTE` is active (or standard RIPA steering via `steer_from_swar_words_core_safe`),
    /// and optionally scales `logits_opt` by `frame.effective_inv_temp`.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline(always)]
    fn steer_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport)> {
        let (frame, report, _frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        Ok((frame, report))
    }

    /// Synthesizes `signal` via [`Self::steer_gateway_signal_safe`], steers `query_tensor` (and
    /// optional `key_tensor` / `logits_opt`), and absorbs the resulting [`GatewayFrame`] sample
    /// into `crystal_record` on $SO(2)^{64}$ using the register-resident `frm_w` words without
    /// reloading `frame.phase_deltas`, returning `(GatewayFrame, CausalProvenanceReport, updated_gravity_mass)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::steer_gateway_signal_safe`].
    #[inline(always)]
    fn steer_and_crystallize_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        crystal_record: &mut CrystallizedAttractorRecord,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(GatewayFrame, CausalProvenanceReport, f32)> {
        let (frame, report, frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        let grav_mass = crystal_record.absorb_gateway_frame_with_words(&frame, &frm_w);
        Ok((frame, report, grav_mass))
    }

    /// Executes the closed-loop **2-Origin Composite Manifold Gateway $\to$ $SO(2)$-Fused
    /// RIPA/PSPM Steering $\to$ Classical Online Toroidal $k$-Means / DP-Means Macro Clustering**
    /// pipeline on `centroids: &mut [CrystallizedAttractorRecord]` in a single zero-allocation call.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`Self::steer_gateway_signal_safe`].
    #[inline(always)]
    fn steer_and_kmeans_crystallize_gateway_signal_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        centroids: &mut [CrystallizedAttractorRecord],
        split_coherence_thresh: f32,
        unresolved_damping_u7: u8,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        f32,
        bool,
    )> {
        let (frame, report, frm_w) = steer_gateway_signal_core_safe(
            self,
            signal,
            trail,
            unresolved_damping_u7,
            query_tensor,
            key_tensor,
            logits_opt,
        )?;
        let (cluster_idx, anchor_id, coh, mass, spawned) =
            CrystallizedAttractorRecord::step_online_toroidal_kmeans_frame_with_words_64(
                centroids,
                &frame,
                &frm_w,
                split_coherence_thresh,
                Some(trail),
            );
        Ok((frame, report, cluster_idx, anchor_id, coh, mass, spawned))
    }

    /// Executes **Coarse-to-Fine Macro-to-Micro Routing ($\mathcal{O}(K) \to \mathcal{O}(M_{\mathrm{local}})$)**
    /// uniting Classical Online Toroidal $k$-Means macro centroids (`macro_centroids`) with local
    /// Bayesian pointer-trail rewiring (`candidate_trails`) and RIPA phase steering.
    ///
    /// When `attention_scores` is `None` and the Coarse step resolves a non-zero `macro_anchor_id`,
    /// the Fine step restricts 64-head coherence and logarithm evaluation to candidates belonging to
    /// `macro_anchor_id` directly on the register-resident `bridged_w`, bypassing out-of-cluster
    /// candidates and eliminating any separate pre-alignment pass over `candidate_trails`.
    ///
    /// Returns `(macro_cluster_idx, macro_anchor_id, macro_coherence, selected_trail_id, bayesian_score, transition_log_prob, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn route_coarse_to_fine_and_steer_bayesian_trails_safe(
        &self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
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
    ) -> GcsoResult<(usize, u32, f32, u32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let (macro_idx, macro_anchor_id, macro_coh, _macro_grav, self_w, win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
            );
        let has_win_centroid =
            macro_idx < macro_centroids.len() && macro_centroids[macro_idx].sample_count > 0;

        let (best_id, out_score, trans_lp, found) =
            route_macro_aligned_bayesian_trails_and_steer_core(
                self_w,
                &win_w,
                has_win_centroid,
                macro_anchor_id,
                0,
                src_trail,
                candidate_trails,
                head_dim,
                num_heads,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                fine_interpolation_u7,
                contradiction_threshold,
                min_step_rad,
                max_rad,
                attention_scores,
                query_tensor,
                key_tensor,
            );

        Ok((
            macro_idx,
            macro_anchor_id,
            macro_coh,
            best_id,
            out_score,
            trans_lp,
            found,
        ))
    }

    /// Executes **Top-2 Liminal Macro-to-Micro Bifurcation Routing ($\mathcal{O}(K) \to \mathcal{O}(M_{\mathrm{local}})$)**
    /// uniting `CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64` with dual-basin
    /// (`aid1 | aid2`) Bayesian pointer-trail rewiring and RIPA phase steering.
    ///
    /// When `is_liminal` is triggered (the top-2 macro centroids lie within `bifurcation_margin`),
    /// `src_trail` is interpolated onto the geodesic midpoint between `aid1` and `aid2` and the
    /// Fine step admits candidates from **both** competing basins (`aid1` and `aid2`).
    ///
    /// Returns `(idx1, idx2, aid1, aid2, is_liminal, selected_trail_id, bayesian_score, transition_log_prob, found_coherent)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, parameters are
    /// non-finite, or tensor dimensions are invalid.
    #[inline(always)]
    fn route_top2_liminal_and_steer_bayesian_trails_safe(
        &self,
        macro_centroids: &[CrystallizedAttractorRecord],
        coarse_pull_u7: u8,
        bifurcation_margin: f32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
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
    ) -> GcsoResult<(usize, usize, u32, u32, bool, u32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || !branchless_is_finite_f32(bifurcation_margin)
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let (idx1, idx2, aid1, aid2, _eff_coh, _tot_grav, is_liminal, self_w, w1, _w2) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_with_words_64(
                macro_centroids,
                src_trail,
                coarse_pull_u7,
                bifurcation_margin,
            );
        let has_win_centroid =
            idx1 < macro_centroids.len() && macro_centroids[idx1].sample_count > 0;
        let secondary_aid = branchless_select_u32(is_liminal, aid2, 0);

        let (best_id, out_score, trans_lp, found) =
            route_macro_aligned_bayesian_trails_and_steer_core(
                self_w,
                &w1,
                has_win_centroid,
                aid1,
                secondary_aid,
                src_trail,
                candidate_trails,
                head_dim,
                num_heads,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                repulsion_penalty,
                fine_interpolation_u7,
                contradiction_threshold,
                min_step_rad,
                max_rad,
                attention_scores,
                query_tensor,
                key_tensor,
            );

        Ok((
            idx1, idx2, aid1, aid2, is_liminal, best_id, out_score, trans_lp, found,
        ))
    }

    /// Folds a zero-copy slice of `.gcso` [`CrystallizedAttractorRecord`]s (`records`) onto
    /// `trail.accumulated_phase_delta` in 8 register-resident `u64` SWAR words weighted by
    /// **Phase-Proximity Gravity** ([`CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64`]),
    /// binds the dominant 32-bit `anchor_id` into `trail.target_anchor_id` and `trail.cluster_id`,
    /// and steers `query_tensor` (and optional `key_tensor`) directly from the folded SWAR words
    /// without constructing an intermediate 128-byte [`GatewayFrame`].
    ///
    /// Returns `(total_gravity_pull, dominant_anchor_id, dominant_ripa_clamp_rad)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `min_step_rad < 0.0`, `max_rad <= 0.0`, or
    /// tensor dimensions are invalid.
    #[inline(always)]
    fn steer_crystallized_ensemble_zero_copy_safe(
        &self,
        records: &[CrystallizedAttractorRecord],
        trail: &mut gcso_pointer_trail_t,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, u32, f32)> {
        if min_step_rad < 0.0 || !branchless_is_finite_f32(min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) =
            validate_closed_loop_tensors(self, max_rad, query_tensor, key_tensor.as_deref())?;

        let init_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force));
        let (folded_w, tot_pull, dom_id, dom_clamp) =
            CrystallizedAttractorRecord::fold_crystallized_ensemble_words_with_init_pull_64(
                records,
                &mut trail.accumulated_phase_delta,
                init_pull,
            );
        if dom_id != 0 {
            trail.target_anchor_id = dom_id;
            trail.cluster_id = dom_id;
        }
        trail.attractor_pull_force = branchless_min_f32(tot_pull, MAX_ATTRACTOR_PULL_FORCE);
        trail.flags |= TRAIL_FLAG_CRYSTALLIZED_CANDIDATE | TRAIL_FLAG_SYNERGY_RESONANCE;

        let eff_ripa = branchless_select_f32(
            branchless_is_finite_f32(dom_clamp) & (dom_clamp > 0.0),
            dom_clamp,
            max_rad,
        );
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let eff_w = filter_qdps_words_64(&folded_w, step_u8);
        if swar_words_nonzero_64(&eff_w) {
            steer_from_swar_words_core_safe(
                &eff_w,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                eff_ripa,
                query_tensor,
                key_tensor,
            );
        }

        Ok((tot_pull, dom_id, eff_ripa))
    }

    /// Executes the full end-to-end **2-Origin Composite Gateway $\to$ 8-Cell SWAR Micro-Swarm $\to$
    /// Hypothesis 1 $\times$ Hypothesis 2 Bayesian Pointer-Trail Selection & Limina Bridging $\to$
    /// Register-Resident $SO(2)$-Fused RIPA/PSPM Steering $\to$ `.gcso` Crystallization** pipeline.
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, SwarmResolutionReport, selected_trail_id, found_coherent_trail)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or `GCSO_RESTRICT` non-aliasing
    /// invariants are violated.
    #[inline(always)]
    fn steer_gateway_swarm_bayesian_and_crystallize_safe<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        candidate_trails_opt: Option<&mut [gcso_pointer_trail_t]>,
        attention_scores_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        step_gain_u7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        unresolved_damping_u7: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        explore_pulse_q7: gcso_q7_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        mut key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        crystal_record_opt: Option<&mut CrystallizedAttractorRecord>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        SwarmResolutionReport,
        u32,
        bool,
    )> {
        let (frame, init_frm_w) = signal.synthesize_gateway_frame_with_words();
        let report = frame.to_causal_report();
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            frame.ripa_clamp_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;

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

        let qdps_step_u8 = frame.qdps_min_step_q7;
        let (swarm_report, mut swar_words) = trail.try_resolve_action_in_swarm_from_words(
            gcso_tagged_ptr_t(frame.tagged_ptr),
            &init_frm_w,
            frame.active_swarm_cell_mask,
            step_gain_u7,
            qdps_step_u8,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        );

        trail.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail.attractor_pull_force, frame.effective_energy),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        let mut selected_trail_id = trail.linked_trail_id;
        let mut found_coherent = false;
        if let Some(candidates) = candidate_trails_opt {
            if !candidates.is_empty() {
                let scores_slice_opt = attention_scores_opt
                    .filter(|s| !s.is_empty())
                    .map(AlignedSliceMut32::as_mut_slice);
                let (best_id, _bayes_score, _mean_coh, _coh_cnt, found_valid, bridged_w, _best_coh) =
                    crate::traits::select_rewire_bridge_and_modulate_with_preloaded_words_64(
                        trail,
                        swar_words,
                        candidates,
                        scores_slice_opt,
                        head_dim as u32,
                        phase_interference_gain,
                        stigmergic_prior_gain,
                        pointer_link_bonus,
                        contradiction_threshold,
                        interpolation_u7,
                        repulsion_penalty,
                        0,
                        None,
                        0.0,
                    );
                selected_trail_id = best_id;
                found_coherent = found_valid;
                swar_words = bridged_w;
                if found_valid {
                    let best_idx = best_id as usize;
                    if best_idx < candidates.len() {
                        let dst_trail = &mut candidates[best_idx];
                        if interpolation_u7 < 128 {
                            *trail_phase_words_mut(dst_trail) = swar_words;
                        }
                        let inherit_anchor = dst_trail.target_anchor_id != 0;
                        trail.target_anchor_id = branchless_select_u32(
                            inherit_anchor,
                            dst_trail.target_anchor_id,
                            trail.target_anchor_id,
                        );
                        if trail.target_anchor_id != 0 {
                            trail.cluster_id = trail.target_anchor_id;
                            dst_trail.target_anchor_id = trail.target_anchor_id;
                            dst_trail.cluster_id = trail.target_anchor_id;
                        }
                    }
                }
            }
        }

        if ((frame.action_mode_flags & ACTION_MODE_LAYER_SKIP) == 0)
            & swarm_report
                .evolved_ptr
                .has_action_mode(ACTION_MODE_DPSR_STEER)
        {
            let clean_damping =
                branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
            let swar_nonzero = swar_words_nonzero_64(&swar_words);
            let use_pspm = ((frame.action_mode_flags & ACTION_MODE_PSPM_ROUTE) != 0)
                & (swarm_report.quorum_mask == 0xFF);
            if use_pspm {
                let is_default_ripa = (explore_pulse_q7 == 0)
                    & swar_nonzero
                    & (frame.ripa_clamp_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits())
                    & (num_heads > 0)
                    & (num_heads <= MAX_ATTENTION_HEADS);
                if is_default_ripa {
                    let rot_dim = ((head_dim >> 2) & !1).max(2).min(head_dim);
                    if let Some(k) = key_tensor.as_deref_mut() {
                        crate::traits::rope::rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
                            query_tensor.as_mut_slice(),
                            k.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                        );
                    } else {
                        crate::traits::rope::rotate_fused_self_ripa_and_pspm_default_words(
                            query_tensor.as_mut_slice(),
                            &swar_words,
                            num_heads,
                            head_dim,
                            rot_dim,
                            true,
                        );
                    }
                } else {
                    let pspm_cfg = frame.to_pspm_config_for_heads(num_heads);
                    dispatch_fused_ripa_and_pspm_words_core(
                        query_tensor,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        &pspm_cfg,
                        &swar_words,
                        &swar_words,
                        &swar_words,
                        frame.ripa_clamp_rad,
                        explore_pulse_q7,
                        head_dim,
                        true,
                    )?;
                    if let Some(k) = key_tensor.as_deref_mut() {
                        if swar_nonzero {
                            steer_from_swar_words_core_safe(
                                &swar_words,
                                swarm_report.quorum_mask,
                                clean_damping,
                                head_dim,
                                num_heads,
                                true,
                                frame.ripa_clamp_rad,
                                k,
                                None,
                            );
                        }
                    }
                }
            } else if swar_nonzero {
                steer_from_swar_words_core_safe(
                    &swar_words,
                    swarm_report.quorum_mask,
                    clean_damping,
                    head_dim,
                    num_heads,
                    true,
                    frame.ripa_clamp_rad,
                    query_tensor,
                    key_tensor,
                );
            }
        }

        if let Some(logits) = logits_opt {
            if !logits.is_empty() {
                scale_f32_slice_unrolled8(logits.as_mut_slice(), frame.effective_inv_temp);
            }
        }

        if let Some(crystal) = crystal_record_opt {
            let _ = crystal.absorb_gateway_frame_with_words(&frame, &swar_words);
        }

        Ok((
            frame,
            report,
            swarm_report,
            selected_trail_id,
            found_coherent,
        ))
    }
}

impl<T: PhaseSteering + PspmRouter + ?Sized> DpsrClosedLoopSteering for T {}

// ===================================================================
// 2. Primary DPSR Phase-Steering & Sub-Head Routing Engine (`16B` Pod)
// ===================================================================

/// Primary Dynamic Phase-Shifted RoPE (DPSR), RIPA, and QDPS execution engine (`16B` aligned, [`Pod`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DpsrEngine {
    /// Dimension per attention head ($d_{\mathrm{head}}$, positive even integer).
    head_dim: u32,
    /// Total number of attention heads ($H > 0$).
    num_heads: u32,
    /// Default RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    ripa_clamp_max_rad: f32,
    /// Default QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    qdps_min_step_rad: f32,
}

// SAFETY: `DpsrEngine` is `#[repr(C, align(16))]` containing four 32-bit primitive fields totaling 16B.
unsafe impl Pod for DpsrEngine {}

const _: () = {
    assert!(size_of::<DpsrEngine>() == 16);
    assert!(align_of::<DpsrEngine>() == 16);
    assert!(offset_of!(DpsrEngine, head_dim) == 0);
    assert!(offset_of!(DpsrEngine, num_heads) == 4);
    assert!(offset_of!(DpsrEngine, ripa_clamp_max_rad) == 8);
    assert!(offset_of!(DpsrEngine, qdps_min_step_rad) == 12);
    assert!(NUM_SWARM_MICRO_CELLS * HEADS_PER_SWARM_CELL == MAX_ATTENTION_HEADS);
};

impl Default for DpsrEngine {
    #[inline(always)]
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_DPSR_HEAD_DIM,
            num_heads: DEFAULT_DPSR_NUM_HEADS,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
        }
    }
}

impl DpsrEngine {
    /// Creates a new [`DpsrEngine`] instance with validated `(head_dim, num_heads)` configuration.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    pub const fn new(head_dim: u32, num_heads: u32) -> GcsoResult<Self> {
        let valid = (head_dim != 0) & ((head_dim & 1) == 0) & (num_heads != 0);
        if !valid {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if (head_dim as usize)
            .checked_mul(num_heads as usize)
            .is_none()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(Self {
            head_dim,
            num_heads,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
        })
    }

    /// Constructs a [`DpsrEngine`] from a C-ABI [`gcso_config_t`] descriptor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// `ripa_clamp_max_rad <= 0.0`, `qdps_min_step_rad < 0.0`, or thresholds are non-finite.
    #[inline(always)]
    pub const fn from_config(config: &gcso_config_t) -> GcsoResult<Self> {
        let mut engine = match Self::new(config.head_dim, config.num_heads) {
            Ok(e) => e,
            Err(err) => return Err(err),
        };
        let valid_ripa =
            branchless_is_finite_f32(config.ripa_clamp_max_rad) & (config.ripa_clamp_max_rad > 0.0);
        let valid_qdps =
            branchless_is_finite_f32(config.qdps_min_step_rad) & (config.qdps_min_step_rad >= 0.0);
        if !(valid_ripa & valid_qdps) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        engine.ripa_clamp_max_rad = config.ripa_clamp_max_rad;
        engine.qdps_min_step_rad = config.qdps_min_step_rad;
        Ok(engine)
    }

    /// Returns the configured RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    #[inline(always)]
    pub const fn ripa_clamp_max_rad(&self) -> f32 {
        self.ripa_clamp_max_rad
    }

    /// Returns the configured QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    pub const fn qdps_min_step_rad(&self) -> f32 {
        self.qdps_min_step_rad
    }

    /// Returns a copy of `self` with an updated RIPA soft-clamping limit (`ripa_clamp_max_rad`).
    #[inline(always)]
    pub const fn with_ripa_clamp_max_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.ripa_clamp_max_rad = branchless_select_f32(valid, max_rad, self.ripa_clamp_max_rad);
        self
    }

    /// Returns a copy of `self` with an updated QDPS step cutoff threshold (`qdps_min_step_rad`).
    #[inline(always)]
    pub const fn with_qdps_min_step_rad(mut self, min_step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(min_step_rad) & (min_step_rad >= 0.0);
        self.qdps_min_step_rad = branchless_select_f32(valid, min_step_rad, self.qdps_min_step_rad);
        self
    }

    /// Returns the total number of `f32` elements expected per single-token Query/Key/Value tensor.
    #[inline(always)]
    pub const fn expected_tensor_len(&self) -> usize {
        (self.head_dim as usize) * (self.num_heads as usize)
    }

    /// Applies pointer-gated 8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering with configurable
    /// `unresolved_damping_u7`, delegating directly to
    /// [`PhaseSteering::apply_pointer_swarm_quorum_gated_steering_safe`].
    ///
    /// # Errors
    /// Propagates any error from [`PhaseSteering::apply_pointer_swarm_quorum_gated_steering_safe`].
    #[inline(always)]
    pub fn apply_swarm_quorum_gated_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        swarm_quorum_mask: u8,
        unresolved_damping_u7: u8,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_pointer_swarm_quorum_gated_steering_safe(
            tagged_ptr,
            swarm_quorum_mask,
            unresolved_damping_u7,
            phase_deltas,
            query_tensor,
            key_tensor,
            max_rad,
        )
    }

    /// Steers `query_tensor` (and optional `key_tensor`) in registers using `self.qdps_min_step_rad`
    /// and `self.ripa_clamp_max_rad` configured on this [`DpsrEngine`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or non-aliasing invariants fail.
    #[inline(always)]
    pub fn steer_qk_with_defaults(
        &self,
        raw_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            self.ripa_clamp_max_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;
        filter_qdps_and_steer_from_phases_64(
            raw_phases,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok(())
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Manifold** into `trail` (updating `trail.target_anchor_id` and
    /// `trail.cluster_id`) and steers `query_tensor` (and optional `key_tensor`) using
    /// `self.qdps_min_step_rad` and `self.ripa_clamp_max_rad`.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or tensor dimensions fail.
    #[inline(always)]
    pub fn steer_unified_field_with_defaults(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            self.ripa_clamp_max_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;
        let (pull, mean_coh, active_cnt, raw_w) = trail
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );
        filter_qdps_and_steer_from_words_64(
            &raw_w,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok((pull, mean_coh, active_cnt))
    }

    /// Executes [`DpsrClosedLoopSteering::select_rewire_and_steer_bayesian_pointer_trail_safe`]
    /// using `self.qdps_min_step_rad` and `self.ripa_clamp_max_rad` configured on this [`DpsrEngine`].
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_coherence, transition_log_prob, found_coherent_transition)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`DpsrClosedLoopSteering::select_rewire_and_steer_bayesian_pointer_trail_safe`].
    #[inline(always)]
    pub fn select_rewire_and_steer_bayesian_with_defaults(
        &self,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(u32, f32, f32, f32, bool)> {
        self.select_rewire_and_steer_bayesian_pointer_trail_safe(
            src_trail,
            candidate_trails,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            attention_scores,
            query_tensor,
            key_tensor,
        )
    }

    /// Executes [`DpsrClosedLoopSteering::steer_and_kmeans_crystallize_gateway_signal_safe`] with
    /// default DP-Means split coherence ([`DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE`] = `0.35`)
    /// and unresolved cell damping (`64` = `0.5x`).
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any error from [`DpsrClosedLoopSteering::steer_and_kmeans_crystallize_gateway_signal_safe`].
    #[inline(always)]
    pub fn steer_and_kmeans_crystallize_with_defaults<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        centroids: &mut [CrystallizedAttractorRecord],
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        f32,
        bool,
    )> {
        self.steer_and_kmeans_crystallize_gateway_signal_safe(
            signal,
            trail,
            centroids,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            64,
            query_tensor,
            key_tensor,
            logits_opt,
        )
    }
}

// ===================================================================
// 3. Minimal Trait Delegations (Inheriting All `rope.rs` Default Implementations)
// ===================================================================

impl PhaseSteering for DpsrEngine {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.num_heads
    }
}

impl EntropyGatedPhaseLookup for DpsrEngine {}

impl DynamicLayerSkipper for DpsrEngine {}

impl PspmRouter for DpsrEngine {}

impl VisualAttentionSteering for DpsrEngine {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::aligned::{AlignedBuffer128, AlignedBuffer32, AlignedBuffer64};
    use crate::traits::branchless::branchless_sincos_q7;
    use crate::traits::gateway::{ObjectiveFact, SubjectivePersona, SystemConstraint};
    use crate::traits::primitives::{
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE,
        ATTRACTOR_SLOT_PERSONA, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    };
    use core::f32::consts::PI;

    #[test]
    fn test_dpsr_engine_creation_aligned_dispatch_and_pod_layout() {
        assert!(DpsrEngine::new(128, 32).is_ok());
        assert!(DpsrEngine::new(127, 32).is_err());
        assert!(DpsrEngine::new(0, 32).is_err());
        assert!(DpsrEngine::new(128, 0).is_err());

        let def = DpsrEngine::default()
            .with_ripa_clamp_max_rad(0.18)
            .with_qdps_min_step_rad(0.01);
        assert_eq!(def.head_dim(), 128);
        assert_eq!(def.num_heads(), 32);
        assert!((def.ripa_clamp_max_rad() - 0.18).abs() < 1e-6);
        assert!((def.qdps_min_step_rad() - 0.01).abs() < 1e-6);
        assert_eq!(def.as_bytes().len(), 16);

        let eng_small = DpsrEngine::new(8, 2).unwrap();
        let mut q128 = AlignedBuffer128::new([
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ]);
        let p16 = AlignedBuffer16::new([64i8; 16]);
        eng_small
            .apply_phase_steering_aligned::<128>(
                &mut q128.as_aligned_slice_mut(),
                &p16.as_aligned_slice(),
            )
            .unwrap();
        assert!((q128.data[0] - 0.0).abs() < 1e-4);
        assert!((q128.data[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_qdps_filter_and_swar_scaling() {
        let engine = DpsrEngine::new(128, 16).unwrap();
        let mut raw_buf =
            AlignedBuffer16::new([1i8, 5, -2, 10, -128, 4, -1, 8, 6, -6, 12, -12, 0, 2, -3, 16]);
        let mut aligned = raw_buf.as_aligned_slice_mut();

        assert!(engine.qdps_filter_step(&mut aligned, 0.1).is_ok());
        let res = aligned.as_slice();
        assert_eq!(res[0], 0);
        assert_eq!(res[1], 5);
        assert_eq!(res[2], 0);
        assert_eq!(res[3], 10);
        assert_eq!(res[4], -128);
        assert_eq!(res[5], 0);
        assert_eq!(res[6], 0);
        assert_eq!(res[7], 8);
        assert_eq!(res[15], 16);

        assert!(engine.scale_phases_swar_u7(&mut aligned, 128).is_ok());
        assert_eq!(aligned.as_slice()[3], 10);
        assert!(engine.scale_phases_swar_u7(&mut aligned, 64).is_ok());
        assert_eq!(aligned.as_slice()[3], 5);
        assert_eq!(aligned.as_slice()[7], 4);
        assert_eq!(aligned.as_slice()[15], 8);
    }

    #[test]
    fn test_lazy_unwrap_override_and_3way_qkv_steering() {
        let engine = DpsrEngine::new(4, 1).unwrap();
        let mut q_buf = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
        let c_buf = AlignedBuffer32::new([PI * 0.5]);

        engine
            .lazy_unwrap_override(&mut q_buf.as_aligned_slice_mut(), &c_buf.as_aligned_slice())
            .unwrap();
        assert!((q_buf.data[0] - 0.0).abs() < 1e-4);
        assert!((q_buf.data[1] - 1.0).abs() < 1e-4);
        assert!((q_buf.data[2] - (-1.0)).abs() < 1e-4);
        assert!((q_buf.data[3] - 0.0).abs() < 1e-4);

        let mut q3 = AlignedBuffer32::new([1.0f32, 0.0, 2.0, 0.0]);
        let mut k3 = AlignedBuffer32::new([0.0f32, 1.0, 0.0, 2.0]);
        let mut v3 = AlignedBuffer32::new([1.0f32, 1.0, 1.0, 1.0]);
        let phases = AlignedBuffer16::new([64i8; 16]);

        engine
            .apply_phase_steering_fused_qkv(
                &mut q3.as_aligned_slice_mut(),
                &mut k3.as_aligned_slice_mut(),
                &mut v3.as_aligned_slice_mut(),
                &phases.as_aligned_slice(),
            )
            .unwrap();
        assert!((q3.data[0] - 0.0).abs() < 1e-4);
        assert!((q3.data[1] - 1.0).abs() < 1e-4);
        assert!((k3.data[0] - (-1.0)).abs() < 1e-4);
        assert!((k3.data[1] - 0.0).abs() < 1e-4);
        assert!((v3.data[0] - (-1.0)).abs() < 1e-4);
        assert!((v3.data[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_slerp_norm_guard_logit_shift_and_procrustes_feedback() {
        let engine = DpsrEngine::new(4, 2).unwrap();
        let mut t_buf = AlignedBuffer32::new([2.0f32; 8]);
        engine
            .slerp_norm_guard_stable(&mut t_buf.as_aligned_slice_mut(), 1.0, 1.0)
            .unwrap();
        let norm_sq: f32 = t_buf.data.iter().map(|v| v * v).sum();
        assert!((norm_sq.sqrt() - 1.0).abs() < 1e-5);

        let mut logits = AlignedBuffer32::new([1.0f32; 8]);
        let mut phases_arr = [0i8; 16];
        phases_arr[0] = 32;
        phases_arr[1] = 64;
        let phases = AlignedBuffer16::new(phases_arr);
        engine
            .fused_logit_shift(
                &mut logits.as_aligned_slice_mut(),
                &phases.as_aligned_slice(),
            )
            .unwrap();
        let expected_shift = (48.0 * PI) / 128.0;
        for &val in &logits.data {
            assert!((val - (1.0 + expected_shift)).abs() < 1e-4);
        }

        let engine_1h = DpsrEngine::new(4, 1).unwrap();
        let mut q_fb = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
        let tgt = AlignedBuffer32::new([0.0f32, 1.0, -1.0, 0.0]);
        let mut trail = gcso_pointer_trail_t::default();
        let fb_deltas = engine_1h
            .steer_trail_with_procrustes_feedback(
                &mut trail,
                &mut q_fb.as_aligned_slice_mut(),
                &tgt.as_aligned_slice(),
                128,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
            )
            .unwrap();
        assert_eq!(fb_deltas.data[0], 64);
        assert_eq!(trail.accumulated_phase_delta[0], 64);
        assert!((q_fb.data[2] - 1.0).abs() > 1e-3);

        let engine_8h = DpsrEngine::new(16, 8).unwrap();
        let mut src_8h = AlignedBuffer32::new([0.0f32; 128]);
        for pair in src_8h.data.chunks_exact_mut(2) {
            pair[0] = 1.0;
            pair[1] = 0.0;
        }
        let mut tgt_8h = src_8h;
        let p_64 = AlignedBuffer16::new([64i8; 16]);
        engine_8h
            .apply_phase_steering(&mut tgt_8h.as_aligned_slice_mut(), &p_64.as_aligned_slice())
            .unwrap();
        let mut out_8h = AlignedBuffer16::new([0i8; 16]);
        engine_8h
            .compute_procrustes_phase_delta(
                &src_8h.as_aligned_slice(),
                &tgt_8h.as_aligned_slice(),
                &mut out_8h.as_aligned_slice_mut(),
            )
            .unwrap();
        for h in 0..8 {
            assert_eq!(out_8h.data[h], 64);
        }

        let (s_pi, c_pi) = branchless_sincos_q7(-128);
        assert!(s_pi.abs() < 1e-6);
        assert!((c_pi - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_seam_free_circular_unwrap_swarm_quorum_unified_field_limina_and_pspm() {
        let engine = DpsrEngine::new(4, 16).unwrap();
        let mut ctx_accum = AlignedBuffer16::new([120i8; 16]);
        let step_deltas = AlignedBuffer16::new([24i8; 16]);

        engine
            .accumulate_context_phases_circular_q7(
                &mut ctx_accum.as_aligned_slice_mut(),
                &step_deltas.as_aligned_slice(),
                0.01,
            )
            .unwrap();
        assert_eq!(ctx_accum.data[0], -112);

        let q_phases = AlignedBuffer16::new([-100i8; 16]);
        let mut q_tensor = AlignedBuffer32::new([1.0f32; 64]);
        let eff = engine
            .apply_seam_free_lazy_unwrap_q7(
                &mut q_tensor.as_aligned_slice_mut(),
                None,
                &q_phases.as_aligned_slice(),
                &ctx_accum.as_aligned_slice(),
                0.01,
                None,
            )
            .unwrap();
        assert_eq!(eff.data[0], 12);

        let ctx_antipodal = AlignedBuffer16::new([27i8; 16]);
        let eff_geo = engine
            .apply_seam_free_lazy_unwrap_q7(
                &mut q_tensor.as_aligned_slice_mut(),
                None,
                &q_phases.as_aligned_slice(),
                &ctx_antipodal.as_aligned_slice(),
                0.0,
                Some(DEFAULT_RIPA_CLAMP_MAX_RAD),
            )
            .unwrap();
        assert_eq!(eff_geo.data[0], -1);

        let ptr = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            0,
        );
        let mut q_swarm = AlignedBuffer32::new([1.0f32; 64]);
        let deltas_64 = AlignedBuffer16::new([64i8; 16]);
        engine
            .apply_swarm_quorum_gated_steering_safe(
                ptr,
                0b0000_0001,
                0,
                &deltas_64.as_aligned_slice(),
                &mut q_swarm.as_aligned_slice_mut(),
                None,
                0.20,
            )
            .unwrap();
        assert!((q_swarm.data[2] - 1.0).abs() > 1e-3);
        assert_eq!(&q_swarm.data[32..36], &[1.0f32; 4]);

        let mut trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x2000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_FULL_ORCHESTRATION,
                0,
            )
            .raw(),
            accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let tgt_64 = AlignedBuffer16::new([32i8; MAX_ATTENTION_HEADS]);
        let mut q_closed = AlignedBuffer32::new([1.0f32; 64]);
        let report = engine
            .resolve_and_apply_swarm_phase_steering_safe(
                &mut trail,
                &tgt_64.as_aligned_slice(),
                128,
                4,
                6,
                0,
                0.10,
                0.15,
                None,
                2,
                128,
                0.01,
                0.20,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(report.hot_path_resolved, 1);
        assert_eq!(report.resolved_cell_count, 8);
        assert_eq!(report.quorum_mask, 0xFF);

        let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [32i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.2f32, 3.0f32];
        let mut uf_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            accumulated_phase_delta: [30i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let uf_report = engine
            .resolve_unified_field_swarm_and_steer_safe(
                &mut uf_trail,
                &unified_anchors,
                &unified_energies,
                None,
                128,
                4,
                6,
                64,
                0.10,
                0.15,
                None,
                2,
                128,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(uf_report.hot_path_resolved, 1);
        assert_eq!(uf_trail.target_anchor_id, 1);
        assert_eq!(uf_trail.cluster_id, 1);

        let mut next_mem_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.0,
            attractor_pull_force: 1.0,
            accumulated_phase_delta: [36i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let (lim_coh, lim_lp, lim_rep) = engine
            .bridge_liminal_trail_and_steer_qk_safe(
                &uf_trail,
                &mut next_mem_trail,
                64,
                0.25,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                2.0,
                Some(&unified_anchors),
                Some(&mut attn_scores.as_aligned_slice_mut()),
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert!(!lim_rep && lim_coh > 0.95 && lim_lp > -0.05);
        assert_eq!(next_mem_trail.target_anchor_id, 1);
        assert!(attn_scores.data[1] > 0.5 && attn_scores.data[0] < -0.5);

        let mut slot_bank = AlignedBuffer64::<
            gcso_q7_t,
            { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS },
        >::splat(32);
        {
            let mut bank_mut = slot_bank.as_aligned_slice_mut();
            let slot_p = bank_mut
                .slot_exact_mut::<MAX_ATTENTION_HEADS>(ATTRACTOR_SLOT_PERSONA as usize)
                .unwrap();
            slot_p[48..64].fill(112);
            let slot_k = bank_mut
                .slot_exact_mut::<MAX_ATTENTION_HEADS>(ATTRACTOR_SLOT_KNOWLEDGE as usize)
                .unwrap();
            slot_k.fill(32);
        }
        trail.accumulated_phase_delta = [30i8; MAX_ATTENTION_HEADS];
        trail.current_ptr = gcso_tagged_ptr_t::pack_action(
            0x3000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_FULL_ORCHESTRATION,
            0,
        )
        .raw();
        let slot_weights = [1.0f32; MAX_ATTRACTOR_DOMAINS];
        let bank_report = engine
            .resolve_slot_bank_swarm_and_steer_safe(
                &mut trail,
                &slot_bank.relax_slice::<16>(),
                &slot_weights,
                DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                None,
                64,
                4,
                7,
                64,
                0.10,
                0.15,
                None,
                2,
                128,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert!(bank_report.resolved_cell_count >= 7);

        let pa = AlignedBuffer16::new([0i8; MAX_ATTENTION_HEADS]);
        let slerped = engine
            .slerp_and_apply_phase_steering_safe(
                &pa.as_aligned_slice(),
                &tgt_64.as_aligned_slice(),
                64,
                0.01,
                0.20,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(slerped.data[0], 16);

        let slerp_t0 = engine
            .slerp_and_apply_phase_steering_safe(
                &pa.as_aligned_slice(),
                &tgt_64.as_aligned_slice(),
                0,
                0.01,
                0.20,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(slerp_t0.data[0], 0);
        let slerp_t128 = engine
            .slerp_and_apply_phase_steering_safe(
                &pa.as_aligned_slice(),
                &tgt_64.as_aligned_slice(),
                128,
                0.01,
                0.20,
                &mut q_closed.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(slerp_t128.data[0], 32);

        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 4,
            num_logic_heads: 8,
            num_explore_heads: 4,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.1,
            ..Default::default()
        };
        let mut q_pspm = AlignedBuffer32::new([1.0f32; 64]);
        engine
            .dispatch_subhead_slot_ensemble_geodesic(
                &mut q_pspm.as_aligned_slice_mut(),
                &pspm_cfg,
                &slot_bank.relax_slice::<16>(),
                0,
                1,
                2,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                8,
                4,
            )
            .unwrap();
        assert!((q_pspm.data[2] - 1.0).abs() > 1e-3);

        let (coh, r_bar, mean_q7, kappa) = engine
            .evaluate_phase_coherence_and_resultant_64(
                &tgt_64.as_aligned_slice(),
                &tgt_64.as_aligned_slice(),
            )
            .unwrap();
        assert!((coh - 1.0).abs() < 1e-4);
        assert!((r_bar - 1.0).abs() < 1e-4);
        assert_eq!(mean_q7, 32);
        assert!(kappa > 10.0);
    }

    #[test]
    fn test_blanket_bayesian_pointer_trail_selection_unified_pspm_and_gateway_crystallization() {
        let engine = DpsrEngine::new(4, 16).unwrap();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x4000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_FULL_ORCHESTRATION,
            0,
        );

        let mut src_trail = gcso_pointer_trail_t {
            current_ptr: ptr.raw(),
            stigmergic_density: 1.5,
            attractor_pull_force: 2.0,
            target_anchor_id: 9,
            cluster_id: 9,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let cand_contradict = gcso_pointer_trail_t {
            current_ptr: 0x4100,
            stigmergic_density: 4.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand_coherent = gcso_pointer_trail_t {
            current_ptr: 0x4200,
            stigmergic_density: 3.0,
            target_anchor_id: 19,
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand_contradict, cand_coherent];
        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let mut q_buf = AlignedBuffer32::new([1.0f32; 64]);
        let mut k_buf = AlignedBuffer32::new([0.5f32; 64]);

        let (sel_id, bayes_score, trans_coh, trans_lp, found) = engine
            .select_rewire_and_steer_bayesian_with_defaults(
                &mut src_trail,
                &mut candidates,
                2.0,
                0.5,
                0.25,
                1.0,
                64,
                0.25,
                Some(&mut attn_scores.as_aligned_slice_mut()),
                &mut q_buf.as_aligned_slice_mut(),
                Some(&mut k_buf.as_aligned_slice_mut()),
            )
            .unwrap();
        assert!(found);
        assert_eq!(sel_id, 1);
        assert_eq!(src_trail.linked_trail_id, 1);
        assert_eq!(src_trail.target_anchor_id, 19);
        assert_eq!(src_trail.cluster_id, 19);
        assert!(bayes_score > 0.5 && trans_coh > 0.95 && trans_lp > -0.05);
        assert!(attn_scores.data[1] > 1.0 && attn_scores.data[0] < -0.5);

        let unified_anchors = [[-96i8; MAX_ATTENTION_HEADS], [30i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.1f32, 3.5f32];
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 4,
            num_logic_heads: 8,
            num_explore_heads: 4,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.2,
            ..Default::default()
        };
        let mut branch_logits =
            AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0, 0.0]);
        let (final_pull, mean_coh, inv_temp, dom_id) = engine
            .steer_unified_field_pspm_and_inv_temperature_safe(
                &mut src_trail,
                &unified_anchors,
                &unified_energies,
                None,
                Some((&pspm_cfg, 8i8)),
                Some((&mut branch_logits.as_aligned_slice_mut(), 1.0)),
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_buf.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(dom_id, 1);
        assert!(final_pull > 7.0 && mean_coh > 0.40);
        assert!(inv_temp > 1.9);
        assert!(branch_logits.data[0] > 1.9);

        let sys_sig = SystemConstraint::new((), 0.20, 0.05, 0xFF);
        let fact_sig = ObjectiveFact::new(
            "RAG fact: zero-allocation Rust kernel",
            3.0,
            0x0024_0000,
            88,
        );
        let pers_sig =
            SubjectivePersona::new("Analytical co-architect persona", 2.0, 0x0000_9000, 12);
        let mut crystal =
            CrystallizedAttractorRecord::new_empty(88, ATTRACTOR_SLOT_KNOWLEDGE, 0x0024_0000);
        let mut gw_logits = AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);

        let (gw_frame, gw_report, grav_mass) = engine
            .steer_and_crystallize_gateway_signal_safe(
                &(&sys_sig, &fact_sig, &pers_sig),
                &mut src_trail,
                &mut crystal,
                64,
                &mut q_buf.as_aligned_slice_mut(),
                Some(&mut k_buf.as_aligned_slice_mut()),
                Some(&mut gw_logits.as_aligned_slice_mut()),
            )
            .unwrap();
        assert_eq!(gw_frame.zimms_section_offset48, 0x0024_0000);
        assert_eq!(src_trail.target_anchor_id, 88);
        assert_eq!(gw_report.dominant_slot, ATTRACTOR_SLOT_KNOWLEDGE);
        assert!(grav_mass > 0.0 && crystal.sample_count == 1 && crystal.verify_integrity());

        let scaled_cfg16 = gw_frame.to_pspm_config_for_heads(16);
        assert_eq!(
            scaled_cfg16.num_fact_heads
                + scaled_cfg16.num_logic_heads
                + scaled_cfg16.num_explore_heads,
            16
        );

        let mut cand_swarm = [cand_contradict, cand_coherent];
        let mut swarm_scores = AlignedBuffer32::new([0.0f32; 8]);
        let (_gw_f2, _gw_r2, sw_rep, sel_swarm_id, found_swarm) = engine
            .steer_gateway_swarm_bayesian_and_crystallize_safe(
                &(&sys_sig, &fact_sig, &pers_sig),
                &mut src_trail,
                Some(&mut cand_swarm),
                Some(&mut swarm_scores.as_aligned_slice_mut()),
                96,
                16,
                6,
                64,
                0.15,
                0.10,
                None,
                3,
                64,
                2.0,
                0.5,
                0.25,
                1.0,
                64,
                -0.25,
                4,
                &mut q_buf.as_aligned_slice_mut(),
                Some(&mut k_buf.as_aligned_slice_mut()),
                Some(&mut gw_logits.as_aligned_slice_mut()),
                Some(&mut crystal),
            )
            .unwrap();
        assert!(sw_rep.quorum_mask != 0);
        assert!(found_swarm);
        assert_eq!(sel_swarm_id, 1);
        assert_eq!(crystal.sample_count, 2);
        assert!(crystal.verify_integrity());

        let mut macro_centroids = [
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(0, ATTRACTOR_SLOT_PERSONA, 0),
        ];
        let fact_cluster_sig =
            ObjectiveFact::new([28i8; MAX_ATTENTION_HEADS], 3.0, 0x0010_0000, 101);
        let (_f_km1, _r_km1, c_idx1, c_id1, _c_coh1, c_mass1, spawned1) = engine
            .steer_and_kmeans_crystallize_with_defaults(
                &fact_cluster_sig,
                &mut src_trail,
                &mut macro_centroids,
                &mut q_buf.as_aligned_slice_mut(),
                None,
                None,
            )
            .unwrap();
        assert!(spawned1);
        assert_eq!(c_idx1, 0);
        assert_eq!(c_id1, 101);
        assert!(c_mass1 > 0.0);

        let persona_cluster_sig =
            SubjectivePersona::new([-84i8; MAX_ATTENTION_HEADS], 2.8, 0x0028_0000, 202);
        let (_f_km2, _r_km2, c_idx2, c_id2, _c_coh2, _c_mass2, spawned2) = engine
            .steer_and_kmeans_crystallize_with_defaults(
                &persona_cluster_sig,
                &mut src_trail,
                &mut macro_centroids,
                &mut q_buf.as_aligned_slice_mut(),
                None,
                None,
            )
            .unwrap();
        assert!(spawned2);
        assert_eq!(c_idx2, 1);
        assert_eq!(c_id2, 202);

        let mut c2f_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.5,
            attractor_pull_force: 1.5,
            ..Default::default()
        };
        let mut c2f_candidates = [
            gcso_pointer_trail_t {
                cluster_id: 101,
                target_anchor_id: 101,
                stigmergic_density: 2.0,
                accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
            gcso_pointer_trail_t {
                cluster_id: 202,
                target_anchor_id: 202,
                stigmergic_density: 2.5,
                accumulated_phase_delta: [-82i8; MAX_ATTENTION_HEADS],
                ..Default::default()
            },
        ];
        let mut c2f_scores = AlignedBuffer32::new([0.0f32; 8]);
        let (m_idx, m_id, m_coh, fine_sel_id, _fine_bs, _fine_lp, fine_found) = engine
            .route_coarse_to_fine_and_steer_bayesian_trails_safe(
                &macro_centroids,
                64,
                &mut c2f_trail,
                &mut c2f_candidates,
                2.0,
                0.5,
                0.35,
                1.0,
                64,
                0.25,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut c2f_scores.as_aligned_slice_mut()),
                &mut q_buf.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(m_idx, 1);
        assert_eq!(m_id, 202);
        assert!(m_coh > 0.95);
        assert!(fine_found);
        assert_eq!(fine_sel_id, 1);
        assert_eq!(c2f_trail.cluster_id, 202);

        let mut c2f_pruned_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [-80i8; MAX_ATTENTION_HEADS],
            stigmergic_density: 1.5,
            attractor_pull_force: 1.5,
            ..Default::default()
        };
        let (mp_idx, mp_id, mp_coh, mp_sel_id, _mp_bs, _mp_lp, mp_found) = engine
            .route_coarse_to_fine_and_steer_bayesian_trails_safe(
                &macro_centroids,
                64,
                &mut c2f_pruned_trail,
                &mut c2f_candidates,
                2.0,
                0.5,
                0.35,
                1.0,
                64,
                0.25,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                None,
                &mut q_buf.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(mp_idx, 1);
        assert_eq!(mp_id, 202);
        assert!(mp_coh > 0.95);
        assert!(mp_found);
        assert_eq!(mp_sel_id, 1);
        assert_eq!(c2f_pruned_trail.cluster_id, 202);

        let (ens_pull, ens_dom_id, _ens_clamp) = engine
            .steer_crystallized_ensemble_zero_copy_safe(
                &macro_centroids,
                &mut c2f_trail,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_buf.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(ens_dom_id, 202);
        assert!(ens_pull > 1.5);
    }
}