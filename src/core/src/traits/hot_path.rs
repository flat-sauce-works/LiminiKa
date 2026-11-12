// name: src/core/src/traits/hot_path.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.2 Micro Observer traits (`traits/hot_path.rs`) for the Geometric Cellular Sheaf
//! Orchestrator (GCSO):
//! - **[`PointerActionHub`]**: $\mathcal{O}(1)$ 64-bit tagged pointer transitions, autonomous
//!   stigmergic mode evolution, pointer inertia (hysteresis anti-thrashing), game-coexistence QoS
//!   pacing, inter-memory Limina bridging (`link_and_bridge_hallucinated_trails`), autonomous
//!   Bayesian pointer-trail selection & rewiring (`select_link_and_bridge_best_next_trail`),
//!   zero-copy candidate-trail Attention score modulation (`modulate_attention_scores_by_candidate_trails`),
//!   single-manifold unified-field resonance & single-scan 8-cell SWAR Micro-Swarm resolution
//!   (preserving 32-bit `target_anchor_id`), and 8-lane working neighborhood synergy.
//! - **[`SwarmCellChunk`]**: Mezzo PagedBlock chunk stepping and inter-layer Skip-Hop Bus
//!   synchronization (both saturating and circular $S^1$ geodesic-folded).
//! - **[`HotPathExecutionEngine`] / [`FractalCascadeCoordinator`] / [`HotPathObserver`]**:
//!   Blanket-implemented micro observers over [`super::ActionHub`] that drive constant-time
//!   $\mathcal{O}(1)$ `Pointer -> DAES -> Bloom Repulsion -> Unified Field / Toroidal k-Means / Limina / Swarm -> QDPS/RIPA/RoPE -> PSPM/SRL`
//!   token cascades with type-level [`BranchlessKernel`] enforcement.

#![deny(unsafe_code)]
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
#![allow(clippy::field_reassign_with_default)]

use super::aligned::{AlignedSlice16, AlignedSlice32, AlignedSliceMut16, AlignedSliceMut32};
use super::branchless::{
    branchless_clamp_f32, branchless_fast_transition_log_prob, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_q7_circular_resultant_64,
    branchless_q7_slerp_64, branchless_rad_to_qdps_step_u8, branchless_relu_f32,
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    branchless_splitmix64, coherence_from_word_arrays_64, load_q7_words_64,
    scale_f32_slice_unrolled8, swar_add_sat_q7_slice_in_place,
    swar_circular_add_and_fold_geodesic_q7_slice_in_place, swar_q7_add_sat_x8,
    swar_slerp_words_with_bias_64, swar_words_equal_64, swar_words_nonzero_64,
    trail_phase_words_mut, trail_phase_words_ref, BranchlessKernel, SWAR_SPLAT_BYTE_64,
};
use super::gateway::CrystallizedAttractorRecord;
use super::primitives::{
    GcsoResult, ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_PASSIVE_SIDECAR, ATTRACTOR_SLOT_ID_MASK,
    ATTRACTOR_SLOT_REPULSION, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS,
};
use super::rope::steer_from_swar_words_core_safe;
use super::telemetry::{
    eval_bitmask_dyadic_ultrametric, select_rewire_bridge_and_modulate_with_preloaded_words_64,
    ContrastivePhaseBloomFilter, PagedBitmaskOps, StigmergicActionPtrOps, StigmergicSwarmOps,
    StigmergicTrailOps, SwarmResolutionReport, DAES_METRIC_FAST_PATH_HIT,
    DAES_METRIC_LAYER_SKIP_COMPENSATED, DAES_METRIC_LAZY_UNWRAP_APPLIED, DAES_METRIC_PHASE_STEERED,
    DAES_METRIC_PSPM_ROUTED, DAES_METRIC_REPULSION_APPLIED, DAES_METRIC_SLOT_SYNERGY_ACTIVE,
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, NUM_SWARM_MICRO_CELLS,
    TRAIL_FLAG_INERTIAL_SKIP_ACTIVE,
    TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
use super::{
    accumulate_q7_slice_into_words_64, fuse_lazy_unwrap_and_qdps_from_words_64, ActionHub,
};
use crate::abi::{
    gcso_daes_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t,
    gcso_srl_descriptor_t, gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT,
};

// ===================================================================
// 1. Action Hub (Sidecar Pointer Table) Core Orchestrator Trait
// ===================================================================

/// Hot-path capability trait for $\mathcal{O}(1)$ tagged pointer transitions, autonomous
/// stigmergic mode orchestration, pointer inertia, game-coexistence QoS pacing, inter-memory
/// Limina bridging, Bayesian pointer-trail selection & rewiring, zero-copy candidate-trail
/// Attention score modulation, single-manifold unified-field resonance, and closed-loop
/// Micro-Swarm action resolution.
///
/// Requires [`BranchlessKernel`] to enforce constant-time execution at the type level.
pub trait PointerActionHub: BranchlessKernel + Send + Sync {
    /// Resets transient pointer trails and slot counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_STATE`] if the Action Hub is locked or uninitialized.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Advances a 64-bit tagged pointer transition in zero-allocation $\mathcal{O}(1)$ mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_ACTION_HUB_FULL`] on failure.
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<()>;

    /// Advances a batch of tagged pointer transitions across multiple tokens in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths differ, or propagates [`Self::step_pointer`] errors.
    #[inline]
    fn step_pointer_batch(
        &mut self,
        current_ptrs: &[u64],
        trails_out: &mut [gcso_pointer_trail_t],
    ) -> GcsoResult<()> {
        if current_ptrs.len() != trails_out.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        for (&ptr, trail) in current_ptrs.iter().zip(trails_out.iter_mut()) {
            self.step_pointer(ptr, trail)?;
        }
        Ok(())
    }

    /// Executes an $\mathcal{O}(1)$ pointer transition, evaluates the [`ContrastivePhaseBloomFilter`],
    /// and autonomously evolves `trail_out.current_ptr` bottom-up while preserving 32-bit
    /// `trail_out.target_anchor_id` in `trail_out.cluster_id`.
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::step_pointer`].
    #[inline]
    fn step_and_orchestrate_mode(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_tagged_ptr_t, u8, bool)> {
        self.step_pointer(input_ptr.raw(), trail_out)?;

        let (density, repulsion_triggered) = bloom_filter.apply_phase_conjugate_repulsion_words(
            super::branchless::trail_phase_words_mut(trail_out),
            repulsion_threshold,
            repulsion_scale_u7,
        );

        // `evolve_trail_mode_stigmergically` synchronizes routing flags with `current_node_ptr`
        let evolved_ptr = trail_out.evolve_trail_mode_stigmergically(
            entropy_h_tilde,
            repulsion_triggered,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
        );
        Ok((evolved_ptr, density, repulsion_triggered))
    }

    /// Advances `input_ptr` with **Pointer Inertia (Hysteresis Anti-Thrashing)** and
    /// **Game-Coexistence QoS Pacing**, keeping `trail_out.flags` synchronized with `final_ptr`
    /// and preserving 32-bit `trail_out.target_anchor_id` in `trail_out.cluster_id` when
    /// `target_anchor_id != 0` and `!repulsion_triggered`.
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::step_pointer`].
    #[inline]
    fn step_and_orchestrate_qos_inertial(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        entropy_h_tilde: f32,
        sliding_flux_phi: f32,
        tau_flux: f32,
        hysteresis_band: f32,
        game_qos_budget_active: bool,
        dominant_attractor_slot: u8,
        low_entropy_skip_thresh: f32,
        high_density_srl_thresh: f32,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(gcso_tagged_ptr_t, u8, bool)> {
        let prev_mode = input_ptr.action_mode_flags();
        let prev_slot = input_ptr.target_attractor_slot();

        let (candidate_ptr, density, repulsion_triggered) = self.step_and_orchestrate_mode(
            input_ptr,
            entropy_h_tilde,
            dominant_attractor_slot,
            low_entropy_skip_thresh,
            high_density_srl_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
            trail_out,
        )?;

        let hyst = branchless_relu_f32(hysteresis_band);
        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            entropy_h_tilde,
            0.0,
        );
        let clean_flux = branchless_select_f32(
            branchless_is_finite_f32(sliding_flux_phi),
            sliding_flux_phi,
            0.0,
        );

        let has_inertia =
            (trail_out.stigmergic_density > 0.0) | (trail_out.attractor_pull_force > 0.0);
        let in_deadband = (clean_h >= branchless_relu_f32(low_entropy_skip_thresh - hyst))
            & (clean_h <= (low_entropy_skip_thresh + hyst))
            & has_inertia
            & (!repulsion_triggered);

        let inertial_ptr = candidate_ptr.with_action_and_slot(prev_slot, prev_mode);
        let after_inertia = gcso_tagged_ptr_t(branchless_select_u64(
            in_deadband,
            inertial_ptr.raw(),
            candidate_ptr.raw(),
        ));

        let flux_surge =
            (clean_flux > (tau_flux + hyst)) | (clean_h > (low_entropy_skip_thresh + 2.0 * hyst));
        let throttle_qos = game_qos_budget_active & (!flux_surge) & (!repulsion_triggered);
        let ambient_mode = branchless_select_u8(
            prev_mode == ACTION_PRESET_PASSIVE_SIDECAR,
            ACTION_PRESET_PASSIVE_SIDECAR,
            ACTION_PRESET_INERTIAL_SKIP,
        );
        let qos_ptr =
            after_inertia.with_action_and_slot(after_inertia.target_attractor_slot(), ambient_mode);
        let final_ptr = gcso_tagged_ptr_t(branchless_select_u64(
            throttle_qos,
            qos_ptr.raw(),
            after_inertia.raw(),
        ));

        trail_out.current_ptr = final_ptr.raw();
        let preserve_unified_anchor = (trail_out.target_anchor_id != 0) & (!repulsion_triggered);
        trail_out.cluster_id = branchless_select_u32(
            preserve_unified_anchor,
            trail_out.target_anchor_id,
            u32::from(final_ptr.target_attractor_slot()),
        );
        trail_out.sync_flags_with_current_ptr();
        Ok((final_ptr, density, repulsion_triggered))
    }

    /// Register-resident variant of [`Self::orchestrate_unified_field_resonance`] accepting
    /// pre-loaded `init_w: [u64; NUM_SWARM_MICRO_CELLS]` and returning
    /// `(final_pull_force, mean_coherence, active_anchor_count, post_words)` in registers (`0` reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline]
    fn orchestrate_unified_field_resonance_from_words(
        &self,
        init_w: [u64; NUM_SWARM_MICRO_CELLS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS])> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some((_, strength)) = repulsion_phases_and_strength {
            if !branchless_is_finite_f32(strength) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        let res = trail.apply_unified_field_anchor_slice_resonance_from_words(
            init_w,
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        );
        trail.sync_flags_with_current_ptr();
        Ok(res)
    }

    /// Orchestrates **Order-Invariant Single-Manifold Unified-Field Resonance** and returns
    /// `(final_pull_force, mean_coherence, active_anchor_count, post_words)` directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline(always)]
    fn orchestrate_unified_field_resonance_to_words(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS])> {
        let init_w = *trail_phase_words_ref(trail);
        self.orchestrate_unified_field_resonance_from_words(
            init_w,
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
            trail,
        )
    }

    /// Orchestrates **Order-Invariant Single-Manifold Unified-Field Resonance** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`) without requiring an 8-slot phase bank or
    /// $8 \times 8$ coupling matrix.
    ///
    /// Automatically records the 32-bit dominant resonant anchor index into
    /// `trail.target_anchor_id` and `trail.cluster_id`, and synchronizes `trail.flags`.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline(always)]
    fn orchestrate_unified_field_resonance(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (pull, coh, cnt, _words) = self.orchestrate_unified_field_resonance_to_words(
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
            trail,
        )?;
        Ok((pull, coh, cnt))
    }

    /// Advances `input_ptr` and resolves the composite phase trajectory of the **Single Unified
    /// Attractor Manifold** (`anchor_phases`, `anchor_base_energies`) across the 8 parallel
    /// 8-head SWAR Micro-Swarm cells ([`StigmergicSwarmOps::resolve_unified_field_in_swarm_to_words`]),
    /// returning both `(SwarmResolutionReport, resolved_swar_words)` in registers (`0` stack reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `qdps_min_step_rad` is negative/non-finite.
    #[inline]
    fn step_and_resolve_in_unified_field_swarm_to_words(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS])> {
        if anchor_phases.len() != anchor_base_energies.len()
            || qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(qdps_min_step_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.step_pointer(input_ptr.raw(), trail_out)?;
        let qdps_step_q7 = branchless_rad_to_qdps_step_u8(qdps_min_step_rad);
        Ok(trail_out.resolve_unified_field_in_swarm_to_words(
            input_ptr,
            anchor_phases,
            anchor_base_energies,
            active_cell_mask,
            step_gain_u7,
            qdps_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        ))
    }

    /// Advances `input_ptr` and resolves the composite phase trajectory of the **Single Unified
    /// Attractor Manifold** (`anchor_phases`, `anchor_base_energies`) across the 8 parallel
    /// 8-head SWAR Micro-Swarm cells ([`StigmergicSwarmOps::resolve_unified_field_in_swarm_with_recruitment`]),
    /// recruiting the already-identified dominant unified-field donor anchor in $\mathcal{O}(1)$
    /// on partial quorum without re-scanning `anchor_phases`, while preserving the full 32-bit
    /// `trail_out.target_anchor_id` in `trail_out.cluster_id`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `qdps_min_step_rad` is negative/non-finite.
    #[inline(always)]
    fn step_and_resolve_in_unified_field_swarm(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<SwarmResolutionReport> {
        Ok(self
            .step_and_resolve_in_unified_field_swarm_to_words(
                input_ptr,
                anchor_phases,
                anchor_base_energies,
                active_cell_mask,
                step_gain_u7,
                qdps_min_step_rad,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?
            .0)
    }

    /// Orchestrates non-linear **Cross-Slot Attractor Synergy** and **Mutual Inhibition** on `trail`
    /// in $\mathcal{O}(1)$ directly from a contiguous 8-slot phase bank (`slot_phase_bank` of length `>= 512`)
    /// with zero data-dependent branches on slot weights.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`]
    /// if bounds or alignment fail.
    #[inline]
    fn orchestrate_cross_slot_synergy_from_bank(
        &self,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        secondary_slot: u8,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32)> {
        let primary_slot = trail.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK;
        let sec_slot = secondary_slot;
        if (sec_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(synergy_gain)
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let coherence = if primary_slot == sec_slot {
            let phases_a =
                slot_phase_bank.slot_exact::<MAX_ATTENTION_HEADS>(usize::from(primary_slot))?;
            let weight_a = branchless_flush_subnormal_f32(branchless_relu_f32(
                slot_weights[usize::from(primary_slot)],
            ));
            let has_weight = weight_a > 1.0e-9;
            if has_weight {
                let acc_w = trail_phase_words_mut(trail);
                let del_w = load_q7_words_64(phases_a);
                acc_w[0] = swar_q7_add_sat_x8(acc_w[0], del_w[0]);
                acc_w[1] = swar_q7_add_sat_x8(acc_w[1], del_w[1]);
                acc_w[2] = swar_q7_add_sat_x8(acc_w[2], del_w[2]);
                acc_w[3] = swar_q7_add_sat_x8(acc_w[3], del_w[3]);
                acc_w[4] = swar_q7_add_sat_x8(acc_w[4], del_w[4]);
                acc_w[5] = swar_q7_add_sat_x8(acc_w[5], del_w[5]);
                acc_w[6] = swar_q7_add_sat_x8(acc_w[6], del_w[6]);
                acc_w[7] = swar_q7_add_sat_x8(acc_w[7], del_w[7]);
            }
            let new_pull = branchless_flush_subnormal_f32(
                branchless_relu_f32(trail.attractor_pull_force) + weight_a,
            );
            trail.attractor_pull_force = branchless_min_f32(new_pull, MAX_ATTRACTOR_PULL_FORCE);
            trail.cluster_id = u32::from(primary_slot);
            branchless_select_f32(has_weight, 1.0, 0.0)
        } else {
            let (phases_a, phases_b) = slot_phase_bank.slot_pair_exact::<MAX_ATTENTION_HEADS>(
                usize::from(primary_slot),
                usize::from(sec_slot),
            )?;
            let weight_a = slot_weights[usize::from(primary_slot)];
            let weight_b = slot_weights[usize::from(sec_slot)];

            let (_syn_pull, coh) = trail.apply_cross_slot_synergy(
                u32::from(primary_slot),
                phases_a,
                weight_a,
                u32::from(sec_slot),
                phases_b,
                weight_b,
                coupling_matrix_8x8,
                synergy_gain,
            );
            coh
        };

        if inhibition_strength > 0.0 {
            let rep_phases = slot_phase_bank
                .slot_exact::<MAX_ATTENTION_HEADS>(usize::from(ATTRACTOR_SLOT_REPULSION))?;
            let rep_weight = slot_weights[usize::from(ATTRACTOR_SLOT_REPULSION)];
            let eff_inh = branchless_select_f32(rep_weight > 1.0e-9, inhibition_strength, 0.0);
            let damped_pull = trail.apply_mutual_slot_inhibition(rep_phases, eff_inh);
            return Ok((damped_pull, coherence));
        }

        Ok((trail.attractor_pull_force, coherence))
    }

    /// Orchestrates multi-slot **Coupled Neighborhood Synergy** and **Mutual Inhibition** starting
    /// from pre-loaded 8-word SWAR registers `acc_w` and returning
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_phase_bank.len() < 512` or gains are non-finite.
    #[inline]
    fn orchestrate_neighborhood_synergy_from_bank_from_words(
        &self,
        acc_w: [u64; NUM_SWARM_MICRO_CELLS],
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_SWARM_MICRO_CELLS])> {
        if !branchless_is_finite_f32(synergy_gain) || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bank_2d =
            slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        Ok(trail.apply_neighborhood_synergy_from_bank_from_words(
            acc_w,
            bank_2d,
            slot_weights,
            coupling_matrix_8x8,
            synergy_gain,
            inhibition_strength,
        ))
    }

    /// Orchestrates multi-slot **Coupled Neighborhood Synergy** and **Mutual Inhibition** on `trail`
    /// in $\mathcal{O}(1)$ zero-copy mode across all active neighbors enabled in `coupling_matrix_8x8`
    /// by delegating directly to [`StigmergicTrailOps::apply_neighborhood_synergy_from_bank`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_phase_bank.len() < 512` or gains are non-finite.
    #[inline(always)]
    fn orchestrate_neighborhood_synergy_from_bank(
        &self,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let acc_w = *trail_phase_words_ref(trail);
        let (pull, coh, neighbors, _words) = self
            .orchestrate_neighborhood_synergy_from_bank_from_words(
                acc_w,
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                synergy_gain,
                inhibition_strength,
                trail,
            )?;
        Ok((pull, coh, neighbors))
    }

    /// Advances `input_ptr` and resolves its target Attractor Slot phase command across the 8
    /// parallel 8-head SWAR Micro-Swarm cells ([`StigmergicSwarmOps::resolve_slot_bank_in_swarm_to_words`]),
    /// returning both `(SwarmResolutionReport, resolved_swar_words)` in registers (`0` stack reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`]
    /// if `slot_phase_bank` has fewer than 512 elements or `qdps_min_step_rad` is negative/non-finite.
    #[inline]
    fn step_and_resolve_in_swarm_to_words(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(SwarmResolutionReport, [u64; NUM_SWARM_MICRO_CELLS])> {
        if qdps_min_step_rad < 0.0 || !branchless_is_finite_f32(qdps_min_step_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bank_2d =
            slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        self.step_pointer(input_ptr.raw(), trail_out)?;
        let qdps_step_q7 = branchless_rad_to_qdps_step_u8(qdps_min_step_rad);
        Ok(trail_out.resolve_slot_bank_in_swarm_to_words(
            input_ptr,
            bank_2d,
            slot_weights,
            coupling_matrix_8x8,
            active_cell_mask,
            step_gain_u7,
            qdps_step_q7,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
        ))
    }

    /// Advances `input_ptr` and resolves its target Attractor Slot phase command across the 8
    /// parallel 8-head SWAR Micro-Swarm cells ([`StigmergicSwarmOps::resolve_slot_bank_in_swarm_with_recruitment`]),
    /// recruiting coupled neighbor slots from `slot_phase_bank` on partial quorum.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`]
    /// if `slot_phase_bank` has fewer than 512 elements or `qdps_min_step_rad` is negative/non-finite.
    #[inline(always)]
    fn step_and_resolve_in_swarm(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<SwarmResolutionReport> {
        Ok(self
            .step_and_resolve_in_swarm_to_words(
                input_ptr,
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                active_cell_mask,
                step_gain_u7,
                qdps_min_step_rad,
                tolerance_q7,
                quorum_threshold,
                entropy_h_tilde,
                low_entropy_skip_thresh,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?
            .0)
    }

    /// Computes direct 256-slot hash index (`0..=255`) for 64-bit pointer trail caching using SplitMix64.
    #[inline(always)]
    #[must_use]
    fn hash_slot256_index(&self, ptr: u64) -> u32 {
        (branchless_splitmix64(ptr) & 0xFF) as u32
    }

    /// Updates cellular state for a trail directly with a Q7 phase delta in fast-path express mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_id` exceeds Action Hub capacity.
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: gcso_q7_t,
    ) -> GcsoResult<()>;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_id` is out of bounds or `pull_force` is non-finite.
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> GcsoResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if either trail ID is out of bounds.
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> GcsoResult<()>;

    /// Links `src_trail_id` ($M_{t-1}$) and `dst_trail_id` ($M_t$) in the Action Hub table via
    /// [`Self::link_hallucinated_trails`] AND bridges the inter-memory **Limina** gap between
    /// `src_trail` and `dst_trail` via [`StigmergicTrailOps::link_with_next_hallucinated_trail`],
    /// evaluating conditional transition log-probability $\ln P(M_t \mid M_{t-1})$ and $H^1$
    /// contradiction repulsion in $\mathcal{O}(1)$ while synchronizing `dst_trail.flags`.
    ///
    /// Returns `(coherence, log_transition_prob, contradiction_repulsed)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::link_hallucinated_trails`] or returns
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `contradiction_threshold` is non-finite.
    #[inline]
    fn link_and_bridge_hallucinated_trails(
        &mut self,
        src_trail_id: u32,
        dst_trail_id: u32,
        src_trail: &mut gcso_pointer_trail_t,
        dst_trail: &mut gcso_pointer_trail_t,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> GcsoResult<(f32, f32, bool)> {
        if !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.link_hallucinated_trails(src_trail_id, dst_trail_id)?;
        let res = src_trail.link_with_next_hallucinated_trail(
            dst_trail_id,
            dst_trail,
            interpolation_u7,
            contradiction_threshold,
        );
        dst_trail.sync_flags_with_current_ptr();
        Ok(res)
    }

    /// Realizes **Stigmergic Pointer-Trail Intelligence (Hypothesis 2 $\times$ Hypothesis 1)**:
    /// scans `candidate_trails` in constant time per candidate via
    /// [`StigmergicTrailOps::select_and_rewire_best_next_trail`] to find the optimal
    /// non-contradictory ($H^1(K; \mathcal{F}) = 0$) next memory trail $j^*$ maximizing the
    /// Bayes-Attention transition score
    /// $\frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t, \boldsymbol{\theta}_j) + \eta \ln(1 + \rho_j) + \lambda_{\mathrm{link}} \mathbb{I}(\mathrm{linked\_trail\_id} == j)$,
    /// registers the pointer link in the Action Hub table via [`Self::link_hallucinated_trails`],
    /// and bridges the inter-memory **Limina** gap on `candidate_trails[j^*]`.
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_coherence, transition_log_prob, found_coherent_transition)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `candidate_trails` is empty, `head_dim == 0`,
    /// `(head_dim & 1) != 0`, or any gain/threshold parameter is non-finite, or propagates errors
    /// from [`Self::link_hallucinated_trails`].
    #[inline]
    fn select_link_and_bridge_best_next_trail(
        &mut self,
        src_trail_id: u32,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
    ) -> GcsoResult<(u32, f32, f32, f32, bool)> {
        if candidate_trails.is_empty()
            || head_dim == 0
            || (head_dim & 1) != 0
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (selected_id, bayes_score, raw_coh, found) = src_trail
            .select_and_rewire_best_next_trail(
                candidate_trails,
                head_dim,
                phase_interference_gain,
                stigmergic_prior_gain,
                pointer_link_bonus,
                contradiction_threshold,
            );

        if !found {
            return Ok((selected_id, 0.0, 0.0, 0.0, false));
        }

        self.link_hallucinated_trails(src_trail_id, selected_id)?;
        let dst_idx = (selected_id as usize).min(candidate_trails.len() - 1);
        let dst_trail = &mut candidate_trails[dst_idx];

        let t_s = u64::from(branchless_select_u8(
            interpolation_u7 > 128,
            128,
            interpolation_u7,
        ));
        if t_s < 128 && raw_coh < 1.0 - 1.0e-7 {
            let src_w = trail_phase_words_ref(src_trail);
            if t_s == 0 {
                *trail_phase_words_mut(dst_trail) = *src_w;
            } else {
                let dst_w = *trail_phase_words_ref(dst_trail);
                let bias_restore = (128u64.wrapping_sub(t_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                *trail_phase_words_mut(dst_trail) =
                    swar_slerp_words_with_bias_64(src_w, &dst_w, t_s, bias_restore);
            }
        }

        let log_prob = branchless_fast_transition_log_prob(raw_coh);

        dst_trail.prev_ptr = src_trail.current_ptr;
        dst_trail.step_count = src_trail.step_count.wrapping_add(1);
        let inherit_anchor = dst_trail.target_anchor_id == 0;
        dst_trail.target_anchor_id = branchless_select_u32(
            inherit_anchor,
            src_trail.target_anchor_id,
            dst_trail.target_anchor_id,
        );
        dst_trail.cluster_id = branchless_select_u32(
            inherit_anchor & (src_trail.target_anchor_id != 0),
            src_trail.target_anchor_id,
            dst_trail.cluster_id,
        );

        let cur_rho =
            branchless_flush_subnormal_f32(branchless_relu_f32(dst_trail.stigmergic_density));
        let pos_rho = cur_rho + 0.25 * (1.0 + branchless_relu_f32(raw_coh));
        dst_trail.stigmergic_density =
            branchless_min_f32(branchless_flush_subnormal_f32(pos_rho), MAX_STIGMERGIC_DENSITY);

        let prev_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(src_trail.attractor_pull_force));
        let cur_pull =
            branchless_flush_subnormal_f32(branchless_relu_f32(dst_trail.attractor_pull_force));
        let bridged_pull =
            cur_pull + 0.5 * prev_pull * branchless_clamp_f32(1.0 + raw_coh, 0.25, 2.0);
        dst_trail.attractor_pull_force = branchless_min_f32(
            branchless_flush_subnormal_f32(bridged_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );

        dst_trail.flags |= branchless_select_u32(raw_coh > 0.25, TRAIL_FLAG_SYNERGY_RESONANCE, 0);
        dst_trail.sync_flags_with_current_ptr();
        Ok((selected_id, bayes_score, raw_coh, log_prob, true))
    }

    /// Directly modulates pre-Softmax candidate memory Attention scores (`attention_scores[..M]`)
    /// in-place in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) constant time per
    /// candidate directly from `candidate_trails: &[gcso_pointer_trail_t]` by delegating to
    /// loop-hoisted [`StigmergicTrailOps::modulate_candidate_trail_scores_in_place`]:
    /// $$\Delta z_{t \to j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t, \boldsymbol{\theta}_j) + \eta \ln(1 + \rho_j) + \lambda_{\mathrm{link}} \mathbb{I}(\mathrm{linked\_trail\_id} == j) - \beta_{\mathrm{rep}} \mathbb{I}(H^1 \neq 0)$$
    ///
    /// Unites **Hypothesis 1** (Joint Probability Chain Rule $\times$ Attention phase interference)
    /// and **Hypothesis 2** (`ActionHub` stigmergic pointer-trail graph state) without copying
    /// phase arrays or densities into intermediate buffers.
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_id, coherent_candidate_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `candidate_trails` is
    /// empty, `head_dim == 0`, `(head_dim & 1) != 0`, or any gain/threshold parameter is non-finite.
    #[inline]
    fn modulate_attention_scores_by_candidate_trails(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        src_trail: &gcso_pointer_trail_t,
        candidate_trails: &[gcso_pointer_trail_t],
        head_dim: u32,
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        contradiction_threshold: f32,
        repulsion_penalty: f32,
    ) -> GcsoResult<(f32, u32, u32)> {
        if attention_scores.is_empty()
            || candidate_trails.is_empty()
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

    /// Performs 4-way unrolled branchless OR bit-tree reduction across 32-byte aligned paged bitmasks.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `bitmasks` is empty.
    #[inline]
    fn reduce_bit_tree(
        &self,
        bitmasks: &AlignedSlice32<'_, gcso_paged_bitmask_t>,
        reduced_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()> {
        if bitmasks.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut b0 = 0u64;
        let mut b1 = 0u64;
        let mut b2 = 0u64;
        let mut b3 = 0u64;

        let mut chunks = bitmasks.as_slice().chunks_exact(4);
        for c in chunks.by_ref() {
            b0 |= (c[0].bits[0] | c[1].bits[0]) | (c[2].bits[0] | c[3].bits[0]);
            b1 |= (c[0].bits[1] | c[1].bits[1]) | (c[2].bits[1] | c[3].bits[1]);
            b2 |= (c[0].bits[2] | c[1].bits[2]) | (c[2].bits[2] | c[3].bits[2]);
            b3 |= (c[0].bits[3] | c[1].bits[3]) | (c[2].bits[3] | c[3].bits[3]);
        }
        for m in chunks.remainder() {
            b0 |= m.bits[0];
            b1 |= m.bits[1];
            b2 |= m.bits[2];
            b3 |= m.bits[3];
        }
        reduced_out.bits = [b0, b1, b2, b3];
        Ok(())
    }

    /// Evaluates branchless SIMD/Warp bitmask reduction over 32-byte aligned PagedBlock KV caches,
    /// partitioning `kv_bits` directly into 64-word lanes (`bits[0..4]`) with 4-way loop unrolling.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `kv_bits` is empty.
    #[inline]
    fn paged_block_warp_bitmask(
        &self,
        kv_bits: &AlignedSlice32<'_, u64>,
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<()> {
        if kv_bits.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut out = gcso_paged_bitmask_t::default();
        let limit = kv_bits.len().min(PAGED_BITMASK_BITS as usize);
        let active_slice = &kv_bits.as_slice()[..limit];

        for (word_slot, word_chunk) in active_slice.chunks(64).enumerate() {
            let mut acc_word = 0u64;
            let mut c4 = word_chunk.chunks_exact(4);
            let mut bit_base = 0u32;
            for q in c4.by_ref() {
                let m0 = ((q[0] != 0) as u64) << bit_base;
                let m1 = ((q[1] != 0) as u64) << (bit_base + 1);
                let m2 = ((q[2] != 0) as u64) << (bit_base + 2);
                let m3 = ((q[3] != 0) as u64) << (bit_base + 3);
                acc_word |= (m0 | m1) | (m2 | m3);
                bit_base += 4;
            }
            for (rem_i, &w) in c4.remainder().iter().enumerate() {
                acc_word |= ((w != 0) as u64) << (bit_base + (rem_i as u32));
            }
            out.bits[word_slot] = acc_word;
        }
        *mask_out = out;
        Ok(())
    }

    /// Returns the maximum slot capacity of the Action Hub table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;
}

// ===================================================================
// 2. Mezzo Swarm Cell Chunk & Skip-Hop Bus Synchronization
// ===================================================================

/// Mezzo-path trait for cellular swarm cell chunk step processing across PagedBlocks (**Mezzo Level**).
///
/// Requires [`BranchlessKernel`] to enforce constant-time execution at the type level.
pub trait SwarmCellChunk: BranchlessKernel + Send + Sync {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    fn step_chunk(&mut self, mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()>;

    /// Synchronizes inter-layer phase deltas across a Mezzo Transformer block group via the Skip-Hop Bus.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are zero or do not match.
    #[inline]
    fn skip_hop_phase_sync(
        &mut self,
        local_phases: &AlignedSlice16<'_, gcso_q7_t>,
        bus_accum_inout: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let len = local_phases.len();
        if len != bus_accum_inout.len() || len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_add_sat_q7_slice_in_place(
            bus_accum_inout.as_mut_slice(),
            local_phases.as_slice(),
            len,
        );
        Ok(())
    }

    /// Synchronizes inter-layer phase deltas across the Skip-Hop Bus using circular $S^1$ addition
    /// (`swar_q7_circular_add_x8`) followed by principal $S^1$ geodesic folding (`swar_q7_fold_s1_geodesic_x8`)
    /// into `[-64, +64]`, preventing multi-layer saturation and $\pm 128$ wrap-around sign flips.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are zero or do not match.
    #[inline]
    fn skip_hop_geodesic_fold_sync(
        &mut self,
        local_phases: &AlignedSlice16<'_, gcso_q7_t>,
        bus_accum_inout: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let len = local_phases.len();
        if len != bus_accum_inout.len() || len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_circular_add_and_fold_geodesic_q7_slice_in_place(
            bus_accum_inout.as_mut_slice(),
            local_phases.as_slice(),
            len,
        );
        Ok(())
    }

    /// Interpolates 64-head Q7 phase trajectories between two Mezzo Transformer block groups on $S^1$.
    #[inline(always)]
    #[must_use]
    fn skip_hop_geodesic_slerp_64(
        &self,
        bus_phases_a: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        local_phases_b: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        t_u7: u8,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        branchless_q7_slerp_64(bus_phases_a, local_phases_b, t_u7)
    }

    /// Steps the PagedBlock chunk and resolves `action_ptr`'s target 64-head phase command across
    /// the 8 Micro-Swarm cells gated by `mask.subswarm_activity_mask_8()`.
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::step_chunk`].
    #[inline]
    fn resolve_chunk_subswarms_masked(
        &mut self,
        mask: &gcso_paged_bitmask_t,
        chunk_len: u32,
        action_ptr: gcso_tagged_ptr_t,
        target_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        step_gain_u7: u8,
        qdps_min_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_inout: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<SwarmResolutionReport> {
        self.step_chunk(mask, chunk_len)?;
        let mask8 = mask.subswarm_activity_mask_8();
        let active_cell_mask = branchless_select_u8(mask8 != 0, mask8, 0xFF);
        Ok(trail_inout.try_resolve_action_in_swarm_masked(
            action_ptr,
            target_phases,
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
        ))
    }

    /// Evaluates the 64-head circular statistics summary `(r_bar, mean_phase_q7, von_mises_kappa)` (Section 4.2).
    #[inline(always)]
    #[must_use]
    fn evaluate_chunk_circular_resultant_64(
        &self,
        chunk_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> (f32, gcso_q7_t, f32) {
        branchless_q7_circular_resultant_64(chunk_phases)
    }

    /// Evaluates stigmergic block overlap count and binary cosine coherence between two PagedBlocks.
    #[inline(always)]
    #[must_use]
    fn evaluate_chunk_stigmergic_coherence(
        &self,
        mask_a: &gcso_paged_bitmask_t,
        mask_b: &gcso_paged_bitmask_t,
    ) -> (u32, f32) {
        (
            mask_a.stigmergic_adjacency_overlap(mask_b),
            mask_a.stigmergic_cosine_similarity(mask_b),
        )
    }

    /// Evaluates the normalized pseudo-ultrametric tree distance $d_{\mathcal{V}}(x, y) \in [0, 1)$ (Lemma 1.1).
    #[inline(always)]
    #[must_use]
    fn evaluate_chunk_dyadic_ultrametric(
        &self,
        mask_a: &gcso_paged_bitmask_t,
        mask_b: &gcso_paged_bitmask_t,
        w_key: f32,
    ) -> f32 {
        eval_bitmask_dyadic_ultrametric(mask_a, mask_b, w_key)
    }
}

// ===================================================================
// 3. Lv.2 Micro Observers: `HotPathExecutionEngine`, `FractalCascadeCoordinator`, & `HotPathObserver`
// ===================================================================

/// Composite trait bundling all Hot-Path micro-capabilities registered on [`ActionHub`]
/// for monomorphic static dispatch with type-level [`BranchlessKernel`] enforcement.
pub trait HotPathExecutionEngine: ActionHub {}

impl<T: ActionHub + ?Sized> HotPathExecutionEngine for T {}

/// Blanket-implemented Lv.2 Micro-Observer coordinator that drives constant-time
/// $\mathcal{O}(1)$ `ActionHub -> DAES -> Bloom Repulsion -> Unified Field / Toroidal k-Means / Limina / Swarm -> QDPS/RIPA/RoPE -> PSPM/SRL`
/// token cascades.
pub trait FractalCascadeCoordinator: HotPathExecutionEngine {
    /// Executes a constant-time per-token stigmergic cascade over [`ActionHub`]:
    /// 1. Validates tensor dimensions, `GCSO_RESTRICT` disjointness, and radian thresholds.
    /// 2. Resolves `input_ptr` against `daes_slot`'s $\mathcal{O}(1)$ shortcut table, steps the
    ///    pointer transition in `trail_out`, and applies [`ContrastivePhaseBloomFilter`] repulsion.
    /// 3. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR QDPS and steers
    ///    `query_tensor` and/or `key_tensor` with RIPA soft-clamping (`ripa_max_rad`).
    /// 4. Pushes the combined telemetry metric code (`DAES_METRIC_*`) into `daes_slot`.
    ///
    /// Returns the observed Bloom repulsion bucket density (`0..=255`).
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`ActionHub::step_hub_with_bloom_repulsion`], [`ActionHub::filter_qdps_and_steer_qk_safe`],
    /// or [`super::telemetry::DaesScratchpad::telemetry_push`].
    #[inline]
    fn execute_stigmergic_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, post_rep_words) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &post_rep_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok(density)
    }

    /// Executes a **Single-Manifold Unified-Field** token cascade without requiring a 512-byte
    /// 8-slot phase bank or $8 \times 8$ coupling matrix:
    /// 1. Steps `input_ptr` in [`ActionHub`] and applies [`ContrastivePhaseBloomFilter`] repulsion.
    /// 2. Folds all active anchors in `anchor_phases` and `anchor_base_energies` on the single
    ///    unified manifold via [`PointerActionHub::orchestrate_unified_field_resonance`], binding
    ///    the 32-bit dominant resonant anchor ID into `trail_out.target_anchor_id` and `trail_out.cluster_id`.
    /// 3. Applies 8-lane SWAR QDPS filtering and RIPA soft-clamped phase steering to `query_tensor`
    ///    and/or `key_tensor`.
    ///
    /// Returns `(repulsion_density, final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::orchestrate_unified_field_resonance`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn execute_unified_field_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, f32, f32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (final_pull, mean_coh, active_anchors, uf_words) = self
            .orchestrate_unified_field_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &uf_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(active_anchors > 0, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok((density, final_pull, mean_coh, active_anchors))
    }

    /// Executes a **Single-Pass $SO(2)$ Group-Closure Unified-Field + PSPM Sub-Head Pseudo-MoE +
    /// Inverse-Temperature Logit Sharpening** token cascade over [`ActionHub`]:
    /// 1. Steps `input_ptr` with `daes_slot` shortcut resolution and `bloom_filter` repulsion.
    /// 2. Folds `anchor_phases` and `anchor_base_energies` on the Single Unified Attractor Manifold.
    /// 3. Applies 8-lane SWAR QDPS step filtering in registers and, when `pspm_config_and_pulse`
    ///    is `Some`, fuses RIPA phase steering and PSPM Fact/Logic/Explore sub-head routing into a
    ///    **single pass** over `query_tensor` directly from 8-word SWAR registers via
    ///    [`ActionHub::filter_qdps_and_steer_qk_with_unified_pspm_safe`] (`0` stack buffer copies).
    /// 4. When `logits_and_temp_gain` is `Some`, modulates vocabulary/branch logits in-place by
    ///    zero-copy 64-head inverse-temperature scaling
    ///    ([`super::rope::PhaseSteering::apply_phase_interference_inv_temperature_scaling_64`]).
    ///
    /// Returns `(repulsion_density, final_pull_force, mean_coherence, applied_inv_temp_scale, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from the underlying `ActionHub` traits.
    #[inline]
    fn execute_unified_field_pspm_and_inv_temp_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        logits_and_temp_gain: Option<(&mut AlignedSliceMut32<'_, f32>, f32)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, f32, f32, f32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (final_pull, mean_coh, active_anchors, raw_words) = self
            .orchestrate_unified_field_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;
        let n_anc = anchor_phases.len();
        let dom_idx = if n_anc > 0 {
            (trail_out.target_anchor_id as usize).min(n_anc - 1)
        } else {
            0
        };
        let single_direct = n_anc == 1
            && repulsion_phases_and_strength.is_none()
            && !swar_words_nonzero_64(&init_w)
            && final_pull > 0.0;
        let fact_w = if single_direct {
            raw_words
        } else if n_anc > 0 {
            load_q7_words_64(&anchor_phases[dom_idx])
        } else {
            [0u64; NUM_SWARM_MICRO_CELLS]
        };
        let fact_eq_raw = single_direct || (n_anc > 0 && swar_words_equal_64(&raw_words, &fact_w));
        let fact_ref = if fact_eq_raw { &raw_words } else { &fact_w };
        let logic_w;
        let logic_ref = if pspm_config_and_pulse.is_some() && n_anc > 1 {
            let l_idx = (dom_idx + 1) % n_anc;
            if l_idx == dom_idx {
                fact_ref
            } else {
                logic_w = load_q7_words_64(&anchor_phases[l_idx]);
                if swar_words_equal_64(&raw_words, &logic_w) {
                    &raw_words
                } else {
                    &logic_w
                }
            }
        } else {
            &raw_words
        };

        let (steered, pspm_routed) = self.steer_qdps_pspm_words_prevalidated_core(
            &raw_words,
            fact_ref,
            logic_ref,
            pspm_config_and_pulse,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        )?;

        let mut inv_temp = 1.0_f32;
        if let Some((logits, temp_gain)) = logits_and_temp_gain {
            if !logits.is_empty() && n_anc > 0 {
                if self.num_heads() as usize == MAX_ATTENTION_HEADS {
                    if !branchless_is_finite_f32(temp_gain) {
                        return Err(GCSO_ERROR_INVALID_ARGUMENT);
                    }
                    let coh = if fact_eq_raw {
                        1.0
                    } else {
                        coherence_from_word_arrays_64(&raw_words, &fact_w)
                    };
                    let applied_temp = branchless_phase_coherence_inv_temp_scale(coh, temp_gain);
                    scale_f32_slice_unrolled8(logits.as_mut_slice(), applied_temp);
                    inv_temp = applied_temp;
                } else {
                    let (_coh, applied_temp) = self
                        .apply_phase_interference_inv_temperature_scaling_64(
                            logits,
                            &trail_out.accumulated_phase_delta,
                            &anchor_phases[dom_idx],
                            temp_gain,
                        )?;
                    inv_temp = applied_temp;
                }
            }
        }

        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(active_anchors > 0, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok((
            density,
            final_pull,
            mean_coh,
            inv_temp,
            trail_out.target_anchor_id,
        ))
    }

    /// Executes a **Coarse-to-Fine Online Toroidal $k$-Means / DP-Means Macro-Centroid** token
    /// cascade ($\mathcal{O}(K)$ 64-bit SWAR) over [`ActionHub`]:
    /// 1. Steps `input_ptr` with `daes_slot` shortcut resolution and `bloom_filter` repulsion.
    /// 2. Routes `trail_out` to the highest-gravity coherent macro centroid $k^*$ in `centroids`
    ///    and steers `query_tensor` and/or `key_tensor` directly from register-resident SWAR words
    ///    via [`ActionHub::route_coarse_to_fine_and_steer_qk_safe`].
    /// 3. Pushes combined DAES telemetry metrics into `daes_slot`.
    ///
    /// Returns `(repulsion_density, selected_cluster_idx, selected_anchor_id, centroid_coherence, gravity_pull)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`].
    #[inline]
    fn execute_coarse_to_fine_kmeans_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        centroids: &[CrystallizedAttractorRecord],
        pull_step_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, usize, u32, f32, f32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (cluster_idx, anchor_id, coh, grav, bridged_w, _win_w) =
            CrystallizedAttractorRecord::route_coarse_to_fine_trail_from_preloaded_words_64(
                centroids,
                trail_out,
                init_w,
                pull_step_u7,
            );
        let steered = self.steer_qdps_words_prevalidated_core(
            &bridged_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(anchor_id != 0, DAES_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok((density, cluster_idx, anchor_id, coh, grav))
    }

    /// Executes a **Top-2 Liminal Bifurcation Macro-Centroid + Dual-Centroid PSPM** token cascade
    /// ($\mathcal{O}(K)$ 64-bit SWAR) over [`ActionHub`]:
    /// 1. Steps `input_ptr` with `daes_slot` shortcut resolution and `bloom_filter` repulsion.
    /// 2. Routes `trail_out` against the Top-2 macro centroids $(k_1^*, k_2^*)$ in `centroids`,
    ///    bridging inter-memory *Limina* boundaries along the shortest $S^1$ geodesic when
    ///    $G_2 \ge \tau_{\mathrm{limina}} G_1$, and steers `query_tensor` / `key_tensor` with
    ///    register-resident PSPM sub-head routing via [`ActionHub::route_top2_liminal_and_steer_qk_safe`].
    /// 3. Pushes combined DAES telemetry metrics into `daes_slot`.
    ///
    /// Returns `(repulsion_density, primary_idx, secondary_idx, primary_anchor_id, secondary_anchor_id, blended_coherence, total_gravity, is_liminal_bifurcation)`.
    ///
    /// # Errors
    /// Propagates any validation or execution error from [`ActionHub`].
    #[inline]
    fn execute_top2_liminal_kmeans_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        centroids: &[CrystallizedAttractorRecord],
        pull_step_u7: u8,
        liminal_gravity_ratio_thresh: f32,
        pspm_config_and_pulse: Option<(&gcso_pspm_config_t, gcso_q7_t)>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, usize, usize, u32, u32, f32, f32, bool)> {
        let has_q = query_tensor.is_some();
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal, raw_w, fact_w, logic_w) =
            CrystallizedAttractorRecord::route_top2_liminal_centroids_from_preloaded_words_64(
                centroids,
                trail_out,
                init_w,
                pull_step_u7,
                liminal_gravity_ratio_thresh,
            );
        let (steered, _pspm_routed) = self.steer_qdps_pspm_words_prevalidated_core(
            &raw_w,
            &fact_w,
            &logic_w,
            pspm_config_and_pulse,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        )?;
        let pspm_routed = has_q & pspm_config_and_pulse.is_some();
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(pspm_routed, DAES_METRIC_PSPM_ROUTED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (aid1 != 0) | is_liminal,
                DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );
        self.telemetry_push(daes_slot, metric)?;
        Ok((
            density, idx1, idx2, aid1, aid2, eff_coh, tot_grav, is_liminal,
        ))
    }

    /// Executes a **Limina Bayesian Pointer-Trail + Attention Phase-Interference** token cascade
    /// uniting **Hypothesis 1** (Persona as Joint Probability Chain Rule $\times$ Attention phase
    /// interference) and **Hypothesis 2** (Intelligence as Stigmergic Pointer-Trail manipulation
    /// in [`ActionHub`]):
    /// 1. Steps `input_ptr` with `daes_slot` shortcut resolution and `bloom_filter` repulsion.
    /// 2. Folds `anchor_phases` and `anchor_base_energies` on the single unified manifold via
    ///    [`PointerActionHub::orchestrate_unified_field_resonance`].
    /// 3. If `prev_memory_trail` is provided, bridges the inter-memory **Limina** gap via
    ///    [`StigmergicTrailOps::bridge_liminal_hallucinated_trail`] to evaluate conditional
    ///    transition log-probability $\ln P(M_t \mid M_{t-1})$ and $H^1$ contradiction repulsion.
    /// 4. If `attention_scores` is provided, modulates candidate memory pre-Softmax Attention
    ///    scores via [`ActionHub::modulate_hub_attention_scores_by_unified_field`].
    /// 5. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR QDPS and steers
    ///    `query_tensor` and/or `key_tensor` with RIPA soft-clamping.
    ///
    /// Returns `(repulsion_density, final_pull_force, limina_transition_log_prob, dominant_anchor_id)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::orchestrate_unified_field_resonance`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn execute_liminal_bayesian_pointer_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        prev_memory_trail: Option<&gcso_pointer_trail_t>,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        attention_interference_gain: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, f32, f32, u32)> {
        if !branchless_is_finite_f32(contradiction_threshold)
            || !branchless_is_finite_f32(attention_interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, mut repulsion_active, step_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (_uf_pull, _mean_coh, active_anchors, mut cur_w) = self
            .orchestrate_unified_field_resonance_from_words(
                step_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;

        let mut limina_log_prob = 0.0_f32;
        if let Some(prev_trail) = prev_memory_trail {
            let (_coh, lp, limina_repulsed) = trail_out.bridge_liminal_hallucinated_trail(
                prev_trail,
                interpolation_u7,
                contradiction_threshold,
            );
            limina_log_prob = lp;
            repulsion_active |= limina_repulsed;
            trail_out.sync_flags_with_current_ptr();
            cur_w = *trail_phase_words_ref(trail_out);
        }

        if let Some(scores) = attention_scores {
            if !anchor_phases.is_empty() && !scores.is_empty() {
                let _ = self.modulate_hub_attention_scores_by_unified_field(
                    scores,
                    trail_out,
                    anchor_phases,
                    attention_interference_gain,
                )?;
            }
        }

        let steered = self.steer_qdps_words_prevalidated_core(
            &cur_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (active_anchors > 0) | prev_memory_trail.is_some(),
                DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );
        self.telemetry_push(daes_slot, metric)?;
        Ok((
            density,
            trail_out.attractor_pull_force,
            limina_log_prob,
            trail_out.target_anchor_id,
        ))
    }

    /// Executes the complete closed-loop **Single-Pass Autonomous Pointer-Trail Selection +
    /// Bayes-Attention Interference** token cascade uniting **Hypothesis 1** (Persona as Joint
    /// Probability Chain Rule $\times$ Attention phase interference) and **Hypothesis 2**
    /// (Intelligence as Stigmergic Pointer-Trail rewiring in [`ActionHub`]) in zero-allocation
    /// ($C_{\mathrm{alloc}} = 0$) mode:
    /// 1. Resolves `input_ptr` against `daes_slot`, steps `trail_out` (identified by `src_trail_id`),
    ///    and applies `bloom_filter` phase-conjugate repulsion.
    /// 2. Folds `anchor_phases` and `anchor_base_energies` on the Single Unified Attractor Manifold
    ///    onto `trail_out` via [`PointerActionHub::orchestrate_unified_field_resonance`].
    /// 3. In a **single pass** over `candidate_trails` via
    ///    [`StigmergicTrailOps::select_rewire_bridge_and_modulate_bayesian_trails`]:
    ///    - Selects the optimal coherent ($H^1(K; \mathcal{F}) = 0$) next memory trail $j^*$,
    ///    - Rewires `trail_out.linked_trail_id = j^*` and links `(src_trail_id, j^*)` in the Action Hub,
    ///    - Bridges the inter-memory **Limina** gap in 8-word SWAR registers, and
    ///    - Simultaneously modulates pre-Softmax `attention_scores` in-place (when provided).
    /// 4. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR QDPS and steers
    ///    `query_tensor` and/or `key_tensor` with RIPA soft-clamping.
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_log_prob, dominant_anchor_id, found_coherent_transition)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::orchestrate_unified_field_resonance`],
    /// [`PointerActionHub::link_hallucinated_trails`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn execute_autonomous_pointer_trail_attention_cascade(
        &mut self,
        src_trail_id: u32,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u32, f32, f32, u32, bool)> {
        if candidate_trails.is_empty()
            || !branchless_is_finite_f32(phase_interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
            || !branchless_is_finite_f32(repulsion_penalty)
            || !branchless_is_finite_f32(contradiction_threshold)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, _density, mut repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (_uf_pull, _mean_coh, active_anchors, pre_bridge_words) = self
            .orchestrate_unified_field_resonance_from_words(
                init_w,
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
                trail_out,
            )?;

        let scores_slice_opt = attention_scores
            .filter(|s| !s.is_empty())
            .map(|s| s.as_mut_slice());

        let (selected_id, bayes_score, _mean_coh, _coh_cnt, found, post_w, best_coh) =
            select_rewire_bridge_and_modulate_with_preloaded_words_64(
                trail_out,
                pre_bridge_words,
                candidate_trails,
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

        let mut log_prob = 0.0_f32;
        if found {
            self.link_hallucinated_trails(src_trail_id, selected_id)?;
            let dst_idx = (selected_id as usize).min(candidate_trails.len() - 1);
            let dst_trail = &mut candidate_trails[dst_idx];
            log_prob = branchless_fast_transition_log_prob(best_coh);
            *trail_phase_words_mut(dst_trail) = post_w;
            let dst_has_anchor = dst_trail.target_anchor_id != 0;
            let inherited_anchor = branchless_select_u32(
                dst_has_anchor,
                dst_trail.target_anchor_id,
                trail_out.target_anchor_id,
            );
            trail_out.target_anchor_id = inherited_anchor;
            dst_trail.target_anchor_id = inherited_anchor;
            if inherited_anchor != 0 {
                trail_out.cluster_id = inherited_anchor;
                dst_trail.cluster_id = inherited_anchor;
            }
            trail_out.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.flags |= TRAIL_FLAG_SYNERGY_RESONANCE | TRAIL_FLAG_LIMINAL_BEAT_ACTIVE;
            dst_trail.sync_flags_with_current_ptr();
        } else {
            repulsion_active = true;
        }
        trail_out.sync_flags_with_current_ptr();

        let steered = self.steer_qdps_words_prevalidated_core(
            &post_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(
                (active_anchors > 0) | found,
                DAES_METRIC_SLOT_SYNERGY_ACTIVE,
                0,
            );
        self.telemetry_push(daes_slot, metric)?;
        Ok((
            selected_id,
            bayes_score,
            log_prob,
            trail_out.target_anchor_id,
            found,
        ))
    }

    /// Executes a **Single-Manifold Unified-Field 8-Cell SWAR Micro-Swarm** token cascade:
    /// resolves the composite phase trajectory of `anchor_phases` and `anchor_base_energies`
    /// across the 8 Micro-Swarm cells via [`PointerActionHub::step_and_resolve_in_unified_field_swarm`]
    /// and steers ONLY the resolved 8-head cell groups (`report.quorum_mask`) in `query_tensor`
    /// and/or `key_tensor`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::step_and_resolve_in_unified_field_swarm`], or
    /// [`super::rope::PhaseSteering::apply_swarm_quorum_gated_steering_safe`].
    #[inline]
    fn execute_unified_field_swarm_token_cascade(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        daes_slot: &mut gcso_daes_slot_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<SwarmResolutionReport> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, effective_ptr) = self.resolve_or_pass_tagged_ptr(daes_slot, input_ptr);
        let (report, resolved_words) = self.step_and_resolve_in_unified_field_swarm_to_words(
            effective_ptr,
            anchor_phases,
            anchor_base_energies,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_rad,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
            trail_out,
        )?;
        let steered = (query_tensor.is_some() | key_tensor.is_some()) & (report.quorum_mask != 0);
        if steered {
            match (query_tensor, key_tensor) {
                (Some(q), k_opt) => {
                    steer_from_swar_words_core_safe(
                        &resolved_words,
                        report.quorum_mask,
                        0,
                        head_dim,
                        num_heads,
                        true,
                        ripa_max_rad,
                        q,
                        k_opt,
                    );
                }
                (None, Some(k)) => {
                    steer_from_swar_words_core_safe(
                        &resolved_words,
                        report.quorum_mask,
                        0,
                        head_dim,
                        num_heads,
                        true,
                        ripa_max_rad,
                        k,
                        None,
                    );
                }
                (None, None) => {}
            }
        }
        let repulsion_active = (trail_out.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | DAES_METRIC_SLOT_SYNERGY_ACTIVE;
        self.telemetry_push(daes_slot, metric)?;
        Ok(report)
    }

    /// Executes a full per-token Hot-Path cascade combining pointer transition, Bloom repulsion,
    /// QDPS/RIPA phase steering, and optional single-pass PSPM sub-head routing + SRL Rank-1
    /// residual injection.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`ActionHub::step_hub_with_bloom_repulsion`], [`ActionHub::filter_qdps_and_steer_qk_safe`],
    /// or [`ActionHub::apply_optional_pspm_and_srl`].
    #[inline]
    fn execute_full_hot_path_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        mut query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        pspm_config: Option<&gcso_pspm_config_t>,
        srl_activation_and_desc: Option<(&AlignedSlice32<'_, f32>, &gcso_srl_descriptor_t)>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, post_rep_words) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &post_rep_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref_mut(),
            key_tensor,
        );
        let extra_metrics = if let Some(q) = query_tensor {
            self.apply_optional_pspm_and_srl(q, head_dim, pspm_config, srl_activation_and_desc)?
        } else {
            0
        };
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | extra_metrics;
        self.telemetry_push(daes_slot, metric)?;
        Ok(density)
    }

    /// Executes a **Lazy Phase Unwrap** token cascade (Section 2.3): fuses relative phase
    /// subtraction (`trail_out.accumulated_phase_delta - context_phases`) and 8-lane SWAR QDPS
    /// filtering directly in 64-bit registers (`fuse_lazy_unwrap_and_qdps_words_64`) and rotates
    /// ONLY `query_tensor` via `steer_from_swar_words_core_safe` (`0` stack buffer allocation,
    /// `0` historical KV cache mutation).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `context_phases.len() < num_heads`, or propagates
    /// errors from [`ActionHub::validate_hub_cascade_params`].
    #[inline]
    fn execute_lazy_unwrap_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            Some(&*query_tensor),
            None,
        )?;
        if context_phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (bypass_hit, _eff_ptr, density, repulsion_active, q_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let swar_words = fuse_lazy_unwrap_and_qdps_from_words_64(
            &q_w,
            &trail_out.accumulated_phase_delta,
            context_phases.as_slice(),
            num_heads,
            qdps_min_step_rad,
            false,
        );
        if swar_words_nonzero_64(&swar_words) {
            steer_from_swar_words_core_safe(
                &swar_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                ripa_max_rad,
                query_tensor,
                None,
            );
        }
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | DAES_METRIC_PHASE_STEERED
            | DAES_METRIC_LAZY_UNWRAP_APPLIED
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok(density)
    }

    /// Executes an **$S^1$ Geodesic Lazy Phase Unwrap** token cascade (Section 2.3): computes
    /// wrap-around-safe circular relative phase differences folded onto `[-64, +64]` + 8-lane SWAR
    /// QDPS filtering directly in 64-bit registers (`fuse_lazy_unwrap_and_qdps_words_64` with
    /// `fold_geodesic = true`) and steers `query_tensor` via `steer_from_swar_words_core_safe`
    /// (`0` stack buffer allocation, `0` historical KV cache mutation).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `context_phases.len() < num_heads`, or propagates
    /// errors from [`ActionHub::validate_hub_cascade_params`].
    #[inline]
    fn execute_geodesic_lazy_unwrap_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            Some(&*query_tensor),
            None,
        )?;
        if context_phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (bypass_hit, _eff_ptr, density, repulsion_active, q_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let swar_words = fuse_lazy_unwrap_and_qdps_from_words_64(
            &q_w,
            &trail_out.accumulated_phase_delta,
            context_phases.as_slice(),
            num_heads,
            qdps_min_step_rad,
            true,
        );
        if swar_words_nonzero_64(&swar_words) {
            steer_from_swar_words_core_safe(
                &swar_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                ripa_max_rad,
                query_tensor,
                None,
            );
        }
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | DAES_METRIC_PHASE_STEERED
            | DAES_METRIC_LAZY_UNWRAP_APPLIED
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok(density)
    }

    /// Executes a **Layer-Skip Phase-Compensated** token cascade (Section 4.1): accumulates
    /// `skipped_layer_deltas` into `trail_out.accumulated_phase_delta` when `skip_layer` is active
    /// (`accumulate_q7_slice_into_words_64`), filters via 8-lane SWAR QDPS, and steers `query_tensor`
    /// and/or `key_tensor`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`ActionHub::step_hub_with_bloom_repulsion`], or [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn execute_layer_skip_compensated_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        skip_layer: bool,
        skipped_layer_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        if skip_layer && skipped_layer_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (bypass_hit, _eff_ptr, density, repulsion_active, mut cur_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        if skip_layer {
            accumulate_q7_slice_into_words_64(
                &mut cur_w,
                &skipped_layer_deltas.as_slice()[..num_heads],
            );
            *trail_phase_words_mut(trail_out) = cur_w;
            trail_out.flags |= TRAIL_FLAG_INERTIAL_SKIP_ACTIVE;
        }
        let steered = self.steer_qdps_words_prevalidated_core(
            &cur_w,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(skip_layer, DAES_METRIC_LAYER_SKIP_COMPENSATED, 0);
        self.telemetry_push(daes_slot, metric)?;
        Ok(density)
    }

    /// Executes an **8-Slot Coupled Neighborhood Synergy** token cascade: steps `input_ptr`,
    /// fuses all active coupled attractor slots from `slot_phase_bank` via
    /// [`PointerActionHub::orchestrate_neighborhood_synergy_from_bank`], applies Bloom repulsion,
    /// and steers `query_tensor` and/or `key_tensor`.
    ///
    /// Returns `(repulsion_density, damped_pull_force, mean_coherence, active_neighbor_count)`.
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::orchestrate_neighborhood_synergy_from_bank`], or
    /// [`ActionHub::filter_qdps_and_steer_qk_safe`].
    #[inline]
    fn execute_slot_synergy_token_cascade(
        &mut self,
        input_ptr: u64,
        daes_slot: &mut gcso_daes_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, f32, f32, u32)> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, init_w) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                daes_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let (damped_pull, mean_coh, active_neighbors, syn_words) = self
            .orchestrate_neighborhood_synergy_from_bank_from_words(
                init_w,
                slot_phase_bank,
                slot_weights,
                coupling_matrix_8x8,
                synergy_gain,
                inhibition_strength,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &syn_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | DAES_METRIC_SLOT_SYNERGY_ACTIVE;
        self.telemetry_push(daes_slot, metric)?;
        Ok((density, damped_pull, mean_coh, active_neighbors))
    }

    /// Executes an **8-Cell SWAR Micro-Swarm Quorum-Gated** token cascade: resolves the target
    /// Attractor Slot phase command across the 8 Micro-Swarm cells via
    /// [`PointerActionHub::step_and_resolve_in_swarm`] and steers ONLY the resolved 8-head cell
    /// groups (`report.quorum_mask`) in `query_tensor` and/or `key_tensor` via
    /// [`super::rope::PhaseSteering::apply_swarm_quorum_gated_steering_safe`].
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`PointerActionHub::step_and_resolve_in_swarm`], or
    /// [`super::rope::PhaseSteering::apply_swarm_quorum_gated_steering_safe`].
    #[inline]
    fn execute_swarm_quorum_token_cascade(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        daes_slot: &mut gcso_daes_slot_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        slot_weights: &[f32; MAX_ATTRACTOR_DOMAINS],
        coupling_matrix_8x8: u64,
        active_cell_mask: u8,
        step_gain_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        tolerance_q7: u8,
        quorum_threshold: u8,
        entropy_h_tilde: f32,
        low_entropy_skip_thresh: f32,
        bloom_filter: Option<&ContrastivePhaseBloomFilter>,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<SwarmResolutionReport> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, effective_ptr) = self.resolve_or_pass_tagged_ptr(daes_slot, input_ptr);
        let (report, resolved_words) = self.step_and_resolve_in_swarm_to_words(
            effective_ptr,
            slot_phase_bank,
            slot_weights,
            coupling_matrix_8x8,
            active_cell_mask,
            step_gain_u7,
            qdps_min_step_rad,
            tolerance_q7,
            quorum_threshold,
            entropy_h_tilde,
            low_entropy_skip_thresh,
            bloom_filter,
            repulsion_threshold,
            repulsion_scale_u7,
            trail_out,
        )?;
        let steered = (query_tensor.is_some() | key_tensor.is_some()) & (report.quorum_mask != 0);
        if steered {
            match (query_tensor, key_tensor) {
                (Some(q), k_opt) => {
                    steer_from_swar_words_core_safe(
                        &resolved_words,
                        report.quorum_mask,
                        0,
                        head_dim,
                        num_heads,
                        true,
                        ripa_max_rad,
                        q,
                        k_opt,
                    );
                }
                (None, Some(k)) => {
                    steer_from_swar_words_core_safe(
                        &resolved_words,
                        report.quorum_mask,
                        0,
                        head_dim,
                        num_heads,
                        true,
                        ripa_max_rad,
                        k,
                        None,
                    );
                }
                (None, None) => {}
            }
        }
        let repulsion_active = (trail_out.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let metric = branchless_select_u8(bypass_hit, DAES_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, DAES_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, DAES_METRIC_REPULSION_APPLIED, 0)
            | DAES_METRIC_SLOT_SYNERGY_ACTIVE;
        self.telemetry_push(daes_slot, metric)?;
        Ok(report)
    }
}

impl<T: ActionHub + ?Sized> FractalCascadeCoordinator for T {}

/// Unified Lv.2 Micro Observer trait over [`ActionHub`] (`traits/hot_path.rs`).
///
/// Observes [`ActionHub`] as a strictly branchless, constant-time $\mathcal{O}(1)$
/// `Pointer -> DAES -> Unified Field / Toroidal k-Means / Limina / Swarm -> RoPE (DPSR/RIPA/QDPS) -> PSPM/SRL`
/// execution pipeline.
pub trait HotPathObserver: HotPathExecutionEngine + FractalCascadeCoordinator {}

impl<T: ActionHub + ?Sized> HotPathObserver for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::gcso_config_t;
    use crate::traits::gateway::DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO;
    use crate::traits::primitives::{
        ACTION_PRESET_EXPRESS_PHASE, ACTION_PRESET_FULL_ORCHESTRATION, ACTION_PRESET_INERTIAL_SKIP,
        ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_SYSTEM_BASELINE,
        DEFAULT_QDPS_MIN_STEP_RAD, DEFAULT_RIPA_CLAMP_MAX_RAD, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    };
    use crate::traits::rope::{
        PhaseSteering, PspmRouter, SrlAdapter, VisualAttentionSteering,
    };
    use crate::traits::telemetry::{
        DaesScratchpad, DEFAULT_SWARM_QUORUM_THRESHOLD, DEFAULT_SWARM_TOLERANCE_Q7,
        TRAIL_FLAG_SYNERGY_RESONANCE,
    };
    use crate::traits::{AlignedBuffer16, AlignedBuffer32, AlignedBuffer64};

    struct MinimalHotPathEngine;

    impl PhaseSteering for MinimalHotPathEngine {
        fn head_dim(&self) -> u32 {
            8
        }

        fn num_heads(&self) -> u32 {
            2
        }
    }

    impl PointerActionHub for MinimalHotPathEngine {
        fn reset(&mut self) -> GcsoResult<()> {
            Ok(())
        }

        fn step_pointer(
            &mut self,
            current_ptr: u64,
            trail_out: &mut gcso_pointer_trail_t,
        ) -> GcsoResult<()> {
            trail_out.record_pointer_transition(current_ptr, 1, 1.8);
            trail_out.accumulate_uniform_phase(24);
            Ok(())
        }

        fn update_cellular_state_express(
            &mut self,
            _trail_id: u32,
            _phase_delta: gcso_q7_t,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn pull_trail_to_attractor(
            &mut self,
            _trail_id: u32,
            _anchor_id: u32,
            _pull_force: f32,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn link_hallucinated_trails(
            &mut self,
            _src_trail_id: u32,
            _dst_trail_id: u32,
        ) -> GcsoResult<()> {
            Ok(())
        }

        fn capacity(&self) -> u32 {
            256
        }

        fn active_count(&self) -> u32 {
            1
        }
    }

    impl PspmRouter for MinimalHotPathEngine {}
    impl SrlAdapter for MinimalHotPathEngine {}
    impl VisualAttentionSteering for MinimalHotPathEngine {}

    impl SwarmCellChunk for MinimalHotPathEngine {
        fn step_chunk(&mut self, _mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()> {
            if chunk_len == 0 {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
    }

    impl DaesScratchpad for MinimalHotPathEngine {
        fn evaluate_auto_tune(
            &self,
            _slot: &gcso_daes_slot_t,
            _config_out: &mut gcso_config_t,
        ) -> GcsoResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_swarm_cell_chunk_skip_hop_and_hot_path_observer_cascades() {
        let mut engine = MinimalHotPathEngine;

        let local_p = AlignedBuffer16::new([20i8; 16]);
        let mut bus_p = AlignedBuffer16::new([12i8; 16]);
        engine
            .skip_hop_phase_sync(
                &local_p.as_aligned_slice(),
                &mut bus_p.as_aligned_slice_mut(),
            )
            .unwrap();
        assert_eq!(bus_p.data[0], 32);

        let wrap_local = AlignedBuffer16::new([20i8; 16]);
        let mut wrap_bus = AlignedBuffer16::new([120i8; 16]);
        engine
            .skip_hop_geodesic_fold_sync(
                &wrap_local.as_aligned_slice(),
                &mut wrap_bus.as_aligned_slice_mut(),
            )
            .unwrap();
        assert!(wrap_bus.data[0].abs() <= 64);

        let mut daes = gcso_daes_slot_t::default();
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut q_buf = AlignedBuffer32::new([1.0f32; 16]);
        let ctx_phases = AlignedBuffer16::new([8i8; 16]);
        let mut trail = gcso_pointer_trail_t::default();
        let ptr = gcso_tagged_ptr_t::pack_action(
            0x1000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_EXPRESS_PHASE,
            16,
        );

        let density = engine
            .execute_geodesic_lazy_unwrap_token_cascade(
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &ctx_phases.as_aligned_slice(),
                &mut q_buf.as_aligned_slice_mut(),
                &mut trail,
            )
            .unwrap();
        assert_eq!(density, 0);
        assert_ne!(
            daes.telemetry_mini_ledger[0] & DAES_METRIC_LAZY_UNWRAP_APPLIED,
            0
        );

        // Verify Single-Manifold Unified-Field token cascade & Limina trail linking
        let unified_anchors = [[0i8; MAX_ATTENTION_HEADS], [24i8; MAX_ATTENTION_HEADS]];
        let unified_energies = [0.0f32, 2.5f32];
        let mut uf_trail = gcso_pointer_trail_t::default();
        let (uf_rep, uf_pull, uf_coh, uf_cnt) = engine
            .execute_unified_field_token_cascade(
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                &unified_anchors,
                &unified_energies,
                None,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut uf_trail,
            )
            .unwrap();
        assert_eq!(uf_rep, 0);
        assert_eq!(uf_cnt, 1);
        assert_eq!(uf_trail.target_anchor_id, 1);
        assert_eq!(uf_trail.cluster_id, 1);
        assert!(uf_pull > 4.0 && (uf_coh - 1.0).abs() < 1e-4);

        // Verify Single-Pass SO(2) Fused Unified-Field + PSPM + Inverse-Temperature Cascade
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 1,
            num_logic_heads: 0,
            num_explore_heads: 1,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.25,
            ..Default::default()
        };
        let mut temp_logits =
            AlignedBuffer32::new([1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25, -0.25]);
        let mut uf_pspm_trail = gcso_pointer_trail_t::default();
        let (_rep_d, pspm_pull, pspm_coh, inv_temp, pspm_dom) = engine
            .execute_unified_field_pspm_and_inv_temp_cascade(
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                &unified_anchors,
                &unified_energies,
                None,
                Some((&pspm_cfg, 8)),
                Some((&mut temp_logits.as_aligned_slice_mut(), 0.75)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut uf_pspm_trail,
            )
            .unwrap();
        assert_eq!(pspm_dom, 1);
        assert!(pspm_pull > 4.0 && (pspm_coh - 1.0).abs() < 1e-4);
        assert!((inv_temp - 1.75).abs() < 1e-4);
        assert!((temp_logits.data[0] - 1.75).abs() < 1e-4);

        // Verify Coarse-to-Fine & Top-2 Liminal k-Means token cascades on HotPathObserver
        let mut centroids = [
            CrystallizedAttractorRecord::new_empty(303, ATTRACTOR_SLOT_KNOWLEDGE, 0),
            CrystallizedAttractorRecord::new_empty(404, ATTRACTOR_SLOT_PERSONA, 0),
        ];
        let _ = centroids[0].absorb_sample_point_64(&[24i8; MAX_ATTENTION_HEADS], 3.0);
        centroids[0].stamp_crc32();
        let _ = centroids[1].absorb_sample_point_64(&[44i8; MAX_ATTENTION_HEADS], 3.0);
        centroids[1].stamp_crc32();
        let mut c2f_trail = gcso_pointer_trail_t::default();
        let (_c2f_rep, c2f_idx, c2f_aid, c2f_coh, c2f_grav) = engine
            .execute_coarse_to_fine_kmeans_token_cascade(
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                &centroids,
                64,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut c2f_trail,
            )
            .unwrap();
        assert_eq!(c2f_idx, 0);
        assert_eq!(c2f_aid, 303);
        assert!(c2f_coh > 0.90 && c2f_grav > 1.0);

        let mut lim_trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [10i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (_lim_rep, idx1, idx2, aid1, aid2, lim_coh, lim_grav, is_lim) = engine
            .execute_top2_liminal_kmeans_token_cascade(
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                &centroids,
                64,
                DEFAULT_LIMINAL_BIFURCATION_GRAVITY_RATIO,
                Some((&pspm_cfg, 4)),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut lim_trail,
            )
            .unwrap();
        assert!(is_lim && idx1 != idx2 && aid1 != 0 && aid2 != 0);
        assert!(lim_coh > 0.80 && lim_grav > 1.0);
        assert_ne!(lim_trail.flags & TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        let mut next_trail = gcso_pointer_trail_t {
            current_ptr: 0x2000,
            accumulated_phase_delta: [28i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let (bridge_coh, bridge_lp, bridge_rep) = engine
            .link_and_bridge_hallucinated_trails(1, 2, &mut uf_trail, &mut next_trail, 64, 0.25)
            .unwrap();
        assert!(!bridge_rep && bridge_coh > 0.95 && bridge_lp > -0.05);
        assert_eq!(uf_trail.linked_trail_id, 2);
        assert_eq!(next_trail.target_anchor_id, 1);

        // Verify Bayesian pointer-trail selection & rewiring + Single-Pass Autonomous Pointer-Trail Attention Cascade
        let cand0 = gcso_pointer_trail_t {
            stigmergic_density: 4.0,
            accumulated_phase_delta: [-96i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let cand1 = gcso_pointer_trail_t {
            stigmergic_density: 3.0,
            accumulated_phase_delta: [26i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let mut candidates = [cand0, cand1];

        let (sel_id, bayes_score, sel_coh, sel_lp, found) = engine
            .select_link_and_bridge_best_next_trail(
                10,
                &mut uf_trail,
                &mut candidates,
                8,
                2.0,
                0.5,
                0.25,
                64,
                0.25,
            )
            .unwrap();
        assert!(found);
        assert_eq!(sel_id, 1);
        assert_eq!(uf_trail.linked_trail_id, 1);
        assert!(bayes_score > 0.5 && sel_coh > 0.95 && sel_lp > -0.05);

        let mut attn_scores = AlignedBuffer32::new([0.0f32; 8]);
        let mut auto_trail = gcso_pointer_trail_t::default();
        let (auto_sel, auto_bayes, auto_lp, auto_dom, auto_found) = engine
            .execute_autonomous_pointer_trail_attention_cascade(
                10,
                ptr.raw(),
                &mut daes,
                &bloom,
                2,
                128,
                &mut candidates,
                2.0,
                0.5,
                0.25,
                1.0,
                64,
                0.25,
                &unified_anchors,
                &unified_energies,
                None,
                Some(&mut attn_scores.as_aligned_slice_mut()),
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                Some(&mut q_buf.as_aligned_slice_mut()),
                None,
                &mut auto_trail,
            )
            .unwrap();
        assert!(auto_found);
        assert_eq!(auto_sel, 1);
        assert_eq!(auto_dom, 1);
        assert!(auto_bayes > 0.5 && auto_lp > -0.05);
        assert!(attn_scores.data[1] > 1.0 && attn_scores.data[0] < -0.5);
    }

    #[test]
    fn test_qos_inertial_orchestration_synergy_and_swarm_resolution() {
        let mut engine = MinimalHotPathEngine;
        let bloom = ContrastivePhaseBloomFilter::new();
        let mut trail = gcso_pointer_trail_t {
            target_anchor_id: 42,
            ..Default::default()
        };

        let init_ptr = gcso_tagged_ptr_t::pack_action(
            0x2000,
            ATTRACTOR_SLOT_PERSONA,
            ACTION_PRESET_INERTIAL_SKIP,
            16,
        );
        let (qos_ptr, _, _) = engine
            .step_and_orchestrate_qos_inertial(
                init_ptr,
                0.22,
                0.05,
                0.25,
                0.05,
                true,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0.15,
                1.50,
                &bloom,
                2,
                128,
                &mut trail,
            )
            .unwrap();
        assert_eq!(qos_ptr.action_mode_flags(), ACTION_PRESET_INERTIAL_SKIP);
        assert_ne!(trail.flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
        assert_eq!(trail.cluster_id, 42);

        let (burst_ptr, _, _) = engine
            .step_and_orchestrate_qos_inertial(
                init_ptr,
                0.45,
                0.50,
                0.25,
                0.05,
                true,
                ATTRACTOR_SLOT_KNOWLEDGE,
                0.15,
                1.50,
                &bloom,
                2,
                128,
                &mut trail,
            )
            .unwrap();
        assert_eq!(
            burst_ptr.action_mode_flags(),
            ACTION_PRESET_FULL_ORCHESTRATION
        );
        assert_eq!(trail.flags & TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, 0);
        assert_eq!(trail.cluster_id, 42);

        let mut bank =
            AlignedBuffer64::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();
        {
            let mut bank_mut = bank.as_aligned_slice_mut();
            let (slot_persona, slot_knowledge) = bank_mut
                .disjoint_slot_pair_exact_mut::<MAX_ATTENTION_HEADS>(
                    usize::from(ATTRACTOR_SLOT_PERSONA),
                    usize::from(ATTRACTOR_SLOT_KNOWLEDGE),
                )
                .unwrap();
            slot_persona.fill(64);
            slot_knowledge.fill(64);
        }

        let weights = [1.0f32, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let bank_slice16 = bank.relax_slice::<16>();
        let (pull, coherence, active_neighbors) = engine
            .orchestrate_neighborhood_synergy_from_bank(
                &bank_slice16,
                &weights,
                DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                0.5,
                0.0,
                &mut trail,
            )
            .unwrap();

        assert_eq!(active_neighbors, 2);
        assert!(coherence > 0.25);
        assert!(pull > 2.0);
        assert_ne!(trail.flags & TRAIL_FLAG_SYNERGY_RESONANCE, 0);

        let mut swarm_trail = gcso_pointer_trail_t::default();
        let mut daes = gcso_daes_slot_t::default();
        let mut q_swarm = AlignedBuffer32::new([1.0f32; 16]);
        let swarm_rep = engine
            .execute_swarm_quorum_token_cascade(
                init_ptr,
                &mut daes,
                &bank_slice16,
                &weights,
                DEFAULT_SLOT_COUPLING_MATRIX_8X8,
                0xFF,
                128,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                DEFAULT_SWARM_TOLERANCE_Q7,
                DEFAULT_SWARM_QUORUM_THRESHOLD,
                0.10,
                0.15,
                None,
                2,
                128,
                Some(&mut q_swarm.as_aligned_slice_mut()),
                None,
                &mut swarm_trail,
            )
            .unwrap();
        assert_eq!(swarm_rep.hot_path_resolved, 1);
        assert_eq!(swarm_rep.quorum_mask, 0xFF);

        // Verify Single-Manifold Unified-Field Swarm cascade
        let uf_anchors = [[64i8; MAX_ATTENTION_HEADS]];
        let uf_energies = [2.0f32];
        let mut uf_swarm_trail = gcso_pointer_trail_t::default();
        let uf_swarm_rep = engine
            .execute_unified_field_swarm_token_cascade(
                init_ptr,
                &mut daes,
                &uf_anchors,
                &uf_energies,
                0xFF,
                128,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                DEFAULT_SWARM_TOLERANCE_Q7,
                DEFAULT_SWARM_QUORUM_THRESHOLD,
                0.10,
                0.15,
                None,
                2,
                128,
                Some(&mut q_swarm.as_aligned_slice_mut()),
                None,
                &mut uf_swarm_trail,
            )
            .unwrap();
        assert_eq!(uf_swarm_rep.hot_path_resolved, 1);
        assert_eq!(uf_swarm_rep.quorum_mask, 0xFF);
        assert_eq!(ATTRACTOR_SLOT_SYSTEM_BASELINE, 0);
    }
}