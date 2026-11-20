// name: src/core/src/stigmergy/attractor/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified phase-manifold attractor field, multi-domain synergy, and persona patching traits.

use crate::foundation::{
    atan2_to_q7_word_x8_simd, branchless_atan2_to_q7_fast, branchless_clamp_f32,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_min_f32, branchless_q7_inhibit_64, branchless_q7_inhibit_words_64,
    branchless_q7_synergy_fuse_64, branchless_q7_synergy_fuse_words_64,
    branchless_q7_unified_field_fold_words_with_init_weight_64, branchless_relu_f32,
    branchless_sincos_q7, branchless_splitmix64, branchless_von_mises_kappa,
    circular_r_bar_from_words_64, coherence_from_word_arrays_with_uniform_flag_64,
    is_word_array_uniform_64, load_q7_words_64, store_q7_words_64, swar_q7_add_sat_x8,
    swar_q7_scale_le128_with_bias_x8, swar_q7_sub_sat_x8, AlignedBuffer16, AlignedSlice16,
    AlignedSlice32, AlignedSliceMut16, BranchlessKernel, ResidualAdapterEngine,
    ATTRACTOR_SLOT_ID_MASK,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_q7, branchless_select_u32, branchless_select_u8,
    gcso_q7_t, GcsoResult, GCSO_ERROR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::attractor::gcso_anchor_type_t;
use crate::stigmergy::trail::{
    gcso_pointer_trail_t, trail_phase_words_mut, trail_phase_words_ref, StigmergicTrailOps,
};
use crate::stigmergy::{
    is_slot_pair_coupled, ATTRACTOR_SLOT_REPULSION, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_SYNERGY_RESONANCE,
};

// ===================================================================
// 5. Unified Macro Attractor Field & 8-Lane Working Space Traits
// ===================================================================

/// Cold-path trait for macro attractor field steering, zero-hyperparameter phase-proximity gravity,
/// and repulsion control on a **Single Unified Phase Manifold ($SO(2)^{d_{\mathrm{head}}/2}$)**
/// without an architectural 8-slot domain restriction (**Macro Level**).
pub trait AttractorField: BranchlessKernel + Send + Sync {
    /// Returns the total number of registered topological anchors (`<= MAX_ANCHORS_PER_FIELD`).
    fn anchor_count(&self) -> u32;

    /// Evaluates evaporative pressure decay across active anchors.
    fn apply_evaporative_decay(&mut self);

    /// Returns the total potential energy accumulated across active anchors in this field.
    fn total_potential_energy(&self) -> f32;

    /// Returns the current accumulator pressure ratio (`active_count / MAX_ANCHORS_PER_FIELD`).
    fn pressure_ratio(&self) -> f32;

    /// Evaporates and removes the weakest non-crystallized anchor slot under accumulator pressure.
    fn evaporate_weakest_anchor(&mut self) -> Option<u32>;

    /// Registers a system prompt anchor in phase space.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `prompt` is empty or `weight` is non-finite.
    /// Returns [`crate::liminal::GCSO_ERROR_OUT_OF_MEMORY`] if all 64 slots are occupied by crystallized anchors.
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32>;

    /// Registers a dense feature embedding vector as a continuous attractor anchor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `embedding` is empty, `anchor_type` is invalid, or `weight` is non-finite.
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32>;

    /// Registers a raw topological anchor point in the attractor field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `dim == 0`, `vec.len() < dim`, or `anchor_type` is invalid.
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32>;

    /// Injects a phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) into the field.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `repulsion_deltas` is empty or `gain` is non-finite.
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()>;

    /// Deterministically projects a UTF-8 prompt string into a 64-head Q7 phase profile using
    /// 8-byte SplitMix64 bijective hashing in $\mathcal{O}(L)$ with zero heap allocation.
    #[inline]
    #[must_use]
    fn hash_prompt_to_phase_profile(
        &self,
        prompt: &str,
        domain_salt: u32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        let bytes = prompt.as_bytes();
        let len = bytes.len();
        let mut seed = 0xCBF2_9CE4_8422_2325_u64 ^ (u64::from(domain_salt) << 32) ^ (len as u64);
        let mut i = 0usize;
        while i + 8 <= len {
            let w = u64::from_le_bytes([
                bytes[i],
                bytes[i + 1],
                bytes[i + 2],
                bytes[i + 3],
                bytes[i + 4],
                bytes[i + 5],
                bytes[i + 6],
                bytes[i + 7],
            ]);
            seed = branchless_splitmix64(seed ^ w.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            i += 8;
        }
        if i < len {
            let mut tail = 0u64;
            let mut shift = 0u32;
            while i < len {
                tail |= u64::from(bytes[i]) << shift;
                shift += 8;
                i += 1;
            }
            seed = branchless_splitmix64(seed ^ tail.wrapping_mul(0xBF58_476D_1CE4_E5B9));
        }
        let mut words = [0u64; 8];
        let mut w = 0usize;
        while w < 8 {
            seed = branchless_splitmix64(
                seed.wrapping_add((w as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
            );
            let shifted = (seed >> 1) & 0x7F7F_7F7F_7F7F_7F7F;
            words[w] = shifted ^ (seed & 0x8080_8080_8080_8080);
            w += 1;
        }
        let mut out = [0i8; MAX_ATTENTION_HEADS];
        store_q7_words_64(&mut out, &words);
        out
    }

    /// Evaluates **Zero-Hyperparameter Unified-Field Superposition** across an arbitrary slice of
    /// anchors (`anchor_phases`, `anchor_base_energies`) on the single phase manifold in strictly
    /// constant time (`0` data-dependent branches on `base_e > 0.0`, `0 fdiv`) via
    /// [`crate::foundation::branchless_q7_unified_field_fold_64`], followed by branchless 8-lane SWAR repulsion subtraction.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`.
    #[inline]
    fn evaluate_unified_field_superposition_from_slices(
        &self,
        trail: &gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_gain: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let ref_w = *trail_phase_words_ref(trail);
        let (mut comp_w, total_pull, _mean_coh, _dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &ref_w,
                0.0,
                anchor_phases,
                anchor_base_energies,
            );

        if let Some((rep_phases, rep_gain)) = repulsion_phases_and_gain {
            let clean_gain = branchless_clamp_f32(branchless_relu_f32(rep_gain), 0.0, 1.0);
            let scale_u7 = (clean_gain * 128.0 + 0.5) as u8;
            if scale_u7 > 0 {
                let s = branchless_select_u8(scale_u7 > 128, 128, scale_u7) as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(0x0101_0101_0101_0101);
                let rep_w = load_q7_words_64(rep_phases);
                let mut c = 0usize;
                while c < 8 {
                    let sr = swar_q7_scale_le128_with_bias_x8(rep_w[c], s, bias_restore);
                    comp_w[c] = swar_q7_sub_sat_x8(comp_w[c], sr);
                    c += 1;
                }
            }
        }
        store_q7_words_64(composite_phase_out, &comp_w);

        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Steers `trail` in-place directly on the **Single Unified Attractor Manifold** and returns
    /// `(final_pull_force, mean_coherence, active_anchor_count, post_words)` in 8 `u64` SWAR registers (`0` reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline]
    fn steer_trail_on_unified_field_from_slices_to_words(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES])> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some((_, strength)) = repulsion_phases_and_strength {
            if !branchless_is_finite_f32(strength) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }
        Ok(trail.apply_unified_field_anchor_slice_resonance_to_words(
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
        ))
    }

    /// Steers `trail` in-place directly on the **Single Unified Attractor Manifold** across an
    /// arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) and optional repulsive
    /// phase inhibitor (`repulsion_phases_and_strength`), automatically binding the dominant
    /// resonant anchor index into `trail.target_anchor_id`.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or `repulsion_phases_and_strength` contains a non-finite strength.
    #[inline(always)]
    fn steer_trail_on_unified_field_from_slices(
        &self,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (pull, coh, cnt, _words) = self.steer_trail_on_unified_field_from_slices_to_words(
            anchor_phases,
            anchor_base_energies,
            repulsion_phases_and_strength,
            trail,
        )?;
        Ok((pull, coh, cnt))
    }

    /// Evaluates composite potential force pull and combines 64-head phase shifts for a given trail.
    ///
    /// # Errors
    /// Returns [`crate::liminal::GCSO_ERROR_INVALID_STATE`] if a custom implementor encounters an uninitialized field.
    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        store_q7_words_64(composite_phase_out, trail_phase_words_ref(trail));
        let safe_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(trail.attractor_pull_force)),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        Ok(safe_pull)
    }

    /// Synthesizes a Product of Lie Group Rotations $\mathbf{R}_{\mathrm{total}} = \prod_{l \in \mathrm{Path}} \mathbf{R}_l(\boldsymbol{\Delta\theta}_l)$
    /// across a hierarchical topic phase tree in $\mathcal{O}(L \cdot H)$ using 8-lane SWAR saturating additions (Section 5.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `hierarchy_levels` is empty.
    #[inline]
    fn compose_lie_group_hierarchy_rotations(
        &self,
        hierarchy_levels: &[&[gcso_q7_t; MAX_ATTENTION_HEADS]],
        total_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if hierarchy_levels.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut acc_w = [0u64; 8];
        for &level_deltas in hierarchy_levels {
            let lw = load_q7_words_64(level_deltas);
            acc_w[0] = swar_q7_add_sat_x8(acc_w[0], lw[0]);
            acc_w[1] = swar_q7_add_sat_x8(acc_w[1], lw[1]);
            acc_w[2] = swar_q7_add_sat_x8(acc_w[2], lw[2]);
            acc_w[3] = swar_q7_add_sat_x8(acc_w[3], lw[3]);
            acc_w[4] = swar_q7_add_sat_x8(acc_w[4], lw[4]);
            acc_w[5] = swar_q7_add_sat_x8(acc_w[5], lw[5]);
            acc_w[6] = swar_q7_add_sat_x8(acc_w[6], lw[6]);
            acc_w[7] = swar_q7_add_sat_x8(acc_w[7], lw[7]);
        }
        store_q7_words_64(total_phase_out, &acc_w);
        Ok(())
    }

    /// Synthesizes a depth-weighted Product of Lie Group Rotations
    /// $\mathbf{R}_{\mathrm{total}} = \prod_{l \in \mathrm{Path}} \mathbf{R}_l(s_l \cdot \boldsymbol{\Delta\theta}_l)$ (Section 5.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `hierarchy_levels` is empty or its length does
    /// not match `level_scales_u7.len()`.
    #[inline]
    fn compose_weighted_lie_group_hierarchy(
        &self,
        hierarchy_levels: &[&[gcso_q7_t; MAX_ATTENTION_HEADS]],
        level_scales_u7: &[u8],
        total_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if hierarchy_levels.is_empty() || hierarchy_levels.len() != level_scales_u7.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut acc_w = [0u64; 8];
        for (&level_deltas, &scale_u7) in hierarchy_levels.iter().zip(level_scales_u7.iter()) {
            if scale_u7 == 0 {
                continue;
            }
            let lw = load_q7_words_64(level_deltas);
            let gt_128 = scale_u7 > 128;
            let s_sub = branchless_select_u8(gt_128, scale_u7.wrapping_sub(128), scale_u7) as u64;
            let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(0x0101_0101_0101_0101);
            let extra_mask = (gt_128 as u64).wrapping_neg();
            let mut c = 0usize;
            while c < 8 {
                let w = lw[c];
                let sw = swar_q7_add_sat_x8(
                    swar_q7_scale_le128_with_bias_x8(w, s_sub, bias_restore),
                    w & extra_mask,
                );
                acc_w[c] = swar_q7_add_sat_x8(acc_w[c], sw);
                c += 1;
            }
        }
        store_q7_words_64(total_phase_out, &acc_w);
        Ok(())
    }

    /// Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors.
    ///
    /// # Errors
    /// Returns [`crate::liminal::GCSO_ERROR_INVALID_STATE`] if crystallization cannot proceed.
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32>;

    /// Clears all registered topological anchors from the attractor field.
    ///
    /// # Errors
    /// Returns [`crate::liminal::GCSO_ERROR_INVALID_STATE`] if the field is locked.
    fn clear_anchors(&mut self) -> GcsoResult<()>;
}

/// Cold-path & Mezzo-path trait managing the optional **8-Lane Hardware Working Cache Window
/// (`MAX_ATTRACTOR_DOMAINS = 8`)** projected from the unified attractor manifold with
/// Zero-Hyperparameter Autonomous Weight Rebalancing, Cross-Lane Synergy, Liminal Beat
/// Perturbation, and Mutual Inhibition (**Macro Space Level**).
pub trait MultiDomainAttractorSpace: BranchlessKernel + Send + Sync {
    /// Advances global step counters across all working lanes and applies evaporative pressure decay.
    fn advance_step(&mut self);

    /// Sets the relative weighting multiplier scalar for a specific working lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `weight` is negative/non-finite.
    fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()>;

    /// Enables or disables a specific working lane (`0..=7`) via bitmask.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()>;

    /// Identifies the dominant working lane (`0..=7`) currently holding maximum potential energy.
    fn find_dominant_domain(&self) -> (u32, f32);

    /// Evaluates unified-field superposition of gravitational forces and composite phase shifts in zero-allocation mode.
    fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32;

    /// Clears all registered topological anchors across all 8 working lanes on the unified field.
    ///
    /// # Errors
    /// Returns [`crate::liminal::GCSO_ERROR_INVALID_STATE`] if any lane accumulator cannot be cleared.
    fn clear_all_domains(&mut self) -> GcsoResult<()>;

    /// Returns the active $8 \times 8$ cross-lane coupling bitmatrix (`u64`).
    #[inline(always)]
    fn slot_coupling_matrix(&self) -> u64 {
        DEFAULT_SLOT_COUPLING_MATRIX_8X8
    }

    /// Evaluates the gravitational pull force and 64-head Q7 phase profile of a single working
    /// lane `domain_id` (`0..=7`) for `trail`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= MAX_ATTRACTOR_DOMAINS` (`8`).
    #[inline]
    fn evaluate_slot_force(
        &self,
        domain_id: u32,
        trail: &gcso_pointer_trail_t,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(self.evaluate_multi_domain_superposition(trail, slot_phase_out))
    }

    /// Projects all 8 Working Lanes (`0..=7`) for `reference_trail` into a contiguous
    /// 512-byte (`[[gcso_q7_t; 64]; 8]`) phase bank (`bank_out`) and bounded weight array (`weights_out`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `bank_out.len() < 512`.
    #[inline]
    fn project_slot_phase_bank(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        bank_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        weights_out: &mut [f32; MAX_ATTRACTOR_DOMAINS],
    ) -> GcsoResult<()> {
        let bank_2d = bank_out.as_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        for slot_id in 0..MAX_ATTRACTOR_DOMAINS {
            let raw_pull =
                self.evaluate_slot_force(slot_id as u32, reference_trail, &mut bank_2d[slot_id])?;
            weights_out[slot_id] = branchless_min_f32(
                branchless_flush_subnormal_f32(branchless_relu_f32(raw_pull)),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }
        Ok(())
    }

    /// Autonomously rebalances all 8 working lane weights (`0..=7`) from each lane's phase
    /// proximity resonance with `reference_trail` (`branchless_q7_resonant_pull_64`) in
    /// division-free (`0 fdiv`) mode, eliminating manual per-domain hyperparameter tuning.
    ///
    /// # Errors
    /// Propagates any error from [`Self::evaluate_slot_force`] or [`Self::set_domain_weight`].
    #[inline]
    fn rebalance_working_lane_weights_from_trail(
        &mut self,
        reference_trail: &gcso_pointer_trail_t,
    ) -> GcsoResult<[f32; MAX_ATTRACTOR_DOMAINS]> {
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        let mut max_pull = 0.0_f32;
        let mut scratch_phase = [0i8; MAX_ATTENTION_HEADS];
        let ref_w = *trail_phase_words_ref(reference_trail);
        let ref_is_uniform = is_word_array_uniform_64(&ref_w);
        let ref_nonzero = if ref_is_uniform {
            ref_w[0] != 0
        } else {
            ((ref_w[0] | ref_w[1])
                | (ref_w[2] | ref_w[3])
                | (ref_w[4] | ref_w[5])
                | (ref_w[6] | ref_w[7]))
                != 0
        };

        for (slot_id, pull_slot) in pulls.iter_mut().enumerate() {
            let raw_pull =
                self.evaluate_slot_force(slot_id as u32, reference_trail, &mut scratch_phase)?;
            let clean_base = branchless_flush_subnormal_f32(branchless_relu_f32(raw_pull));
            let res_pull = if clean_base > 0.0 {
                let coh = if ref_nonzero {
                    let lane_w = load_q7_words_64(&scratch_phase);
                    coherence_from_word_arrays_with_uniform_flag_64(&ref_w, ref_is_uniform, &lane_w)
                } else {
                    0.0
                };
                let proximity_gain = branchless_clamp_f32(1.0 + coh, 0.25, 2.0);
                branchless_flush_subnormal_f32(clean_base * proximity_gain)
            } else {
                0.0
            };
            *pull_slot = res_pull;
            max_pull = branchless_select_f32(res_pull > max_pull, res_pull, max_pull);
        }

        let mut weights = [1.0_f32; MAX_ATTRACTOR_DOMAINS];
        let has_max = max_pull > 1.0e-6;
        let inv_max = branchless_inv_pos_f32(branchless_select_f32(has_max, max_pull, 1.0));

        for (slot_id, (&p, weight_slot)) in pulls
            .iter()
            .zip(weights.iter_mut())
            .enumerate()
            .take(MAX_ATTRACTOR_DOMAINS)
        {
            let norm_w = branchless_clamp_f32(0.25 + 1.75 * (p * inv_max), 0.25, 2.0);
            let w = branchless_select_f32(has_max & (p > 0.0), norm_w, 1.0);
            self.set_domain_weight(slot_id as u32, w)?;
            *weight_slot = w;
        }
        Ok(weights)
    }

    /// Evaluates unified-field superposition across the 8 working lanes by projecting the 8-lane
    /// phase bank, applying mutual inhibition from Lane 3 (`ATTRACTOR_SLOT_REPULSION`), and
    /// folding all coupled lanes via [`branchless_q7_synergy_fuse_64`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline]
    fn evaluate_unified_superposition_from_working_bank(
        &self,
        trail: &gcso_pointer_trail_t,
        synergy_gain: f32,
        inhibition_strength: f32,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if !branchless_is_finite_f32(synergy_gain) || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut bank =
            AlignedBuffer16::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        self.project_slot_phase_bank(trail, &mut bank.as_aligned_slice_mut(), &mut pulls)?;
        let bank_2d = bank
            .as_aligned_slice_mut()
            .into_slot_bank_mut::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;

        let rep_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
        let rep_pull = pulls[rep_idx];
        let rep_w = load_q7_words_64(&bank_2d[rep_idx]);
        let apply_rep = (rep_pull > 1.0e-6) & (inhibition_strength > 0.0);

        let mut running_w = [0u64; 8];
        let mut running_pull = 0.0_f32;
        let mut primary_slot = 0u32;
        let coupling = self.slot_coupling_matrix();

        for idx in 0..MAX_ATTRACTOR_DOMAINS {
            if idx == rep_idx || pulls[idx] <= 0.0 {
                continue;
            }
            let mut lane_w = load_q7_words_64(&bank_2d[idx]);
            let mut lane_pull = pulls[idx];
            if apply_rep {
                lane_pull = branchless_q7_inhibit_words_64(
                    &mut lane_w,
                    lane_pull,
                    &rep_w,
                    inhibition_strength,
                );
            }
            if running_pull <= 0.0 {
                running_w = lane_w;
                running_pull = lane_pull;
                primary_slot = idx as u32;
            } else {
                let coupled = is_slot_pair_coupled(coupling, primary_slot, idx as u32);
                let eff_gain = branchless_select_f32(coupled, synergy_gain, 0.0);
                let (fused_w, new_pull, _coh) = branchless_q7_synergy_fuse_words_64(
                    &running_w,
                    running_pull,
                    &lane_w,
                    lane_pull,
                    eff_gain,
                );
                running_w = fused_w;
                running_pull = new_pull;
            }
        }

        if running_pull <= 0.0 {
            store_q7_words_64(composite_phase_out, trail_phase_words_ref(trail));
            return Ok(0.0);
        }

        store_q7_words_64(composite_phase_out, &running_w);
        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Evaluates non-linear **Cross-Slot Synergy** and **Liminal Beat Interference** between
    /// two working lanes (`slot_a`, `slot_b`) gated by `coupling_matrix_8x8` (`u64`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `slot_a >= 8`, `slot_b >= 8`, or `synergy_gain` is non-finite.
    #[inline]
    fn evaluate_cross_slot_synergy(
        &self,
        slot_a: u32,
        slot_b: u32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        trail: &gcso_pointer_trail_t,
        synergy_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, f32)> {
        if (slot_a as usize) >= MAX_ATTRACTOR_DOMAINS
            || (slot_b as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(synergy_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut phase_a = [0i8; MAX_ATTENTION_HEADS];
        let mut phase_b = [0i8; MAX_ATTENTION_HEADS];
        let pull_a = self.evaluate_slot_force(slot_a, trail, &mut phase_a)?;
        let pull_b = self.evaluate_slot_force(slot_b, trail, &mut phase_b)?;

        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, slot_a, slot_b);
        let effective_gain = branchless_select_f32(coupled, synergy_gain, 0.0);

        let (raw_pull, coherence) = branchless_q7_synergy_fuse_64(
            &phase_a,
            pull_a,
            &phase_b,
            pull_b,
            effective_gain,
            synergy_phase_out,
        );
        let bounded_pull = branchless_min_f32(
            branchless_flush_subnormal_f32(raw_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        Ok((bounded_pull, coherence))
    }

    /// Evaluates branchless **Mutual Inhibition** on `target_slot` by an opposing `inhibitor_slot`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `target_slot >= 8`, `inhibitor_slot >= 8`,
    /// or `inhibition_strength` is non-finite.
    #[inline]
    fn evaluate_mutual_slot_inhibition(
        &self,
        target_slot: u32,
        inhibitor_slot: u32,
        inhibition_strength: f32,
        trail: &gcso_pointer_trail_t,
        inhibited_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        if (target_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || (inhibitor_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let pull_target = self.evaluate_slot_force(target_slot, trail, inhibited_phase_out)?;
        let mut inhibitor_phase = [0i8; MAX_ATTENTION_HEADS];
        let pull_inhibitor =
            self.evaluate_slot_force(inhibitor_slot, trail, &mut inhibitor_phase)?;

        let eff_inhibition =
            branchless_select_f32(pull_inhibitor > 1.0e-9, inhibition_strength, 0.0);
        let damped = branchless_q7_inhibit_64(
            inhibited_phase_out,
            pull_target,
            &inhibitor_phase,
            eff_inhibition,
        );
        Ok(branchless_min_f32(
            branchless_flush_subnormal_f32(damped),
            MAX_ATTRACTOR_PULL_FORCE,
        ))
    }

    /// Steers `trail` in-place by extracting the primary Target Working Lane ID (`0..=7`)
    /// from `trail.current_ptr`, fusing it synergistically with `secondary_slot`, and applying
    /// mutual inhibition from [`ATTRACTOR_SLOT_REPULSION`] (`lane 3`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `secondary_slot >= 8` or gains are non-finite.
    #[inline]
    fn steer_trail_by_pointer_slot_synergy(
        &self,
        secondary_slot: u32,
        coupling_matrix_8x8: u64,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32)> {
        if (secondary_slot as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(synergy_gain)
            || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let primary_slot = u32::from(trail.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK);
        let coupled = is_slot_pair_coupled(coupling_matrix_8x8, primary_slot, secondary_slot);
        let effective_gain = branchless_select_f32(coupled, synergy_gain, 0.0);

        let mut phase_a = [0i8; MAX_ATTENTION_HEADS];
        let pull_a = self.evaluate_slot_force(primary_slot, trail, &mut phase_a)?;
        let wa = load_q7_words_64(&phase_a);

        let (syn_pull, coherence, syn_w) = if primary_slot == secondary_slot {
            let (sw, raw_p, coh) =
                branchless_q7_synergy_fuse_words_64(&wa, pull_a, &wa, pull_a, effective_gain);
            (
                branchless_min_f32(
                    branchless_flush_subnormal_f32(raw_p),
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                coh,
                sw,
            )
        } else {
            let mut phase_b = [0i8; MAX_ATTENTION_HEADS];
            let pull_b = self.evaluate_slot_force(secondary_slot, trail, &mut phase_b)?;
            let wb = load_q7_words_64(&phase_b);
            let (sw, raw_p, coh) =
                branchless_q7_synergy_fuse_words_64(&wa, pull_a, &wb, pull_b, effective_gain);
            (
                branchless_min_f32(
                    branchless_flush_subnormal_f32(raw_p),
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                coh,
                sw,
            )
        };

        let mut tw = *trail_phase_words_ref(trail);
        tw[0] = swar_q7_add_sat_x8(tw[0], syn_w[0]);
        tw[1] = swar_q7_add_sat_x8(tw[1], syn_w[1]);
        tw[2] = swar_q7_add_sat_x8(tw[2], syn_w[2]);
        tw[3] = swar_q7_add_sat_x8(tw[3], syn_w[3]);
        tw[4] = swar_q7_add_sat_x8(tw[4], syn_w[4]);
        tw[5] = swar_q7_add_sat_x8(tw[5], syn_w[5]);
        tw[6] = swar_q7_add_sat_x8(tw[6], syn_w[6]);
        tw[7] = swar_q7_add_sat_x8(tw[7], syn_w[7]);

        let raw_pull = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail.attractor_pull_force) + syn_pull,
        );
        let pull_out = branchless_min_f32(raw_pull, MAX_ATTRACTOR_PULL_FORCE);

        let is_resonant = coupled & (coherence > 0.25) & (syn_pull > 0.0);
        let is_liminal = coupled & (coherence < -0.10) & (syn_pull > 0.0);
        let flags_out = trail.flags
            | branchless_select_u32(is_resonant, TRAIL_FLAG_SYNERGY_RESONANCE, 0)
            | branchless_select_u32(is_liminal, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE, 0);

        *trail_phase_words_mut(trail) = tw;
        trail.attractor_pull_force = pull_out;
        trail.flags = flags_out;

        if inhibition_strength > 0.0 {
            let mut rep_phase = [0i8; MAX_ATTENTION_HEADS];
            let rep_pull = self.evaluate_slot_force(
                u32::from(ATTRACTOR_SLOT_REPULSION),
                trail,
                &mut rep_phase,
            )?;
            let eff_inh = branchless_select_f32(rep_pull > 1.0e-9, inhibition_strength, 0.0);
            let damped = trail.apply_mutual_slot_inhibition(&rep_phase, eff_inh);
            return Ok((damped, coherence));
        }

        Ok((pull_out, coherence))
    }

    /// Steers `trail` in-place across the entire coupled neighborhood of its embedded primary
    /// Target Working Lane ID (`0..=7`) and returns
    /// `(damped_pull_force, mean_coherence, active_neighbor_count, post_words)` in 8 `u64` SWAR registers (`0` reload).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline]
    fn steer_trail_by_coupled_neighborhood_to_words(
        &self,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32, [u64; NUM_STIGMERGIC_GRANULES])> {
        if !branchless_is_finite_f32(synergy_gain) || !branchless_is_finite_f32(inhibition_strength)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut bank =
            AlignedBuffer16::<gcso_q7_t, { MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS }>::zeroed();
        let mut weights = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        self.project_slot_phase_bank(trail, &mut bank.as_aligned_slice_mut(), &mut weights)?;
        let bank_2d = bank
            .as_aligned_slice()
            .as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        Ok(trail.apply_neighborhood_synergy_from_bank_to_words(
            bank_2d,
            &weights,
            self.slot_coupling_matrix(),
            synergy_gain,
            inhibition_strength,
        ))
    }

    /// Steers `trail` in-place across the entire coupled neighborhood of its embedded primary
    /// Target Working Lane ID (`0..=7`) by projecting all 8 lanes and delegating to
    /// [`StigmergicTrailOps::apply_neighborhood_synergy_from_bank`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `synergy_gain` or `inhibition_strength` is non-finite.
    #[inline(always)]
    fn steer_trail_by_coupled_neighborhood(
        &self,
        synergy_gain: f32,
        inhibition_strength: f32,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, u32)> {
        let (pull, coh, neighbors, _words) = self.steer_trail_by_coupled_neighborhood_to_words(
            synergy_gain,
            inhibition_strength,
            trail,
        )?;
        Ok((pull, coh, neighbors))
    }
}

/// Cold-path trait for dynamic persona phase modulation patches without altering base weights (**Macro Level**).
pub trait PersonaPatcher: BranchlessKernel + Send + Sync {
    /// Applies a binary persona phase modulation patch to the runtime context.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `patch_data` is empty or malformed.
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()>;

    /// Executes **Persistent Topological Mode Quantization** (Section 3.3) in row-major order,
    /// returning `Ok((0.0, 0.0))` with zeroed `profile_out` if total trail weight is zero.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `coherent_trails` is empty or `min_coherence_r_bar` is non-finite.
    #[inline]
    fn quantize_persistent_topological_mode(
        &self,
        coherent_trails: &[gcso_pointer_trail_t],
        min_coherence_r_bar: f32,
        profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, f32)> {
        if coherent_trails.is_empty() || !branchless_is_finite_f32(min_coherence_r_bar) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut candidate = [0i8; MAX_ATTENTION_HEADS];
        let candidate_words: [u64; 8];
        if coherent_trails.len() == 1 {
            let trail = &coherent_trails[0];
            let d_bounded =
                branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
            let f_bounded =
                branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
            let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
            if w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            candidate = trail.accumulated_phase_delta;
            candidate_words = *trail_phase_words_ref(trail);
        } else {
            let mut sum_s = [[0.0_f32; 8]; 8];
            let mut sum_c = [[0.0_f32; 8]; 8];
            let mut word_s = [0.0_f32; 8];
            let mut word_c = [0.0_f32; 8];
            let mut total_w = 0.0_f32;
            let mut all_byte_uniform = true;
            let mut all_word_uniform = true;
            let mut uniform_s = 0.0_f32;
            let mut uniform_c = 0.0_f32;

            for trail in coherent_trails {
                let d_bounded =
                    branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
                let f_bounded =
                    branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
                let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
                if w > 0.0 {
                    total_w += w;
                    let tw = *trail_phase_words_ref(trail);
                    let w0 = tw[0];
                    if is_word_array_uniform_64(&tw) {
                        if w0 == w0.rotate_right(8) {
                            let (s, c) = branchless_sincos_q7(w0 as gcso_q7_t);
                            uniform_s += w * s;
                            uniform_c += w * c;
                        } else {
                            all_byte_uniform = false;
                            let b = w0.to_le_bytes();
                            let mut lane = 0usize;
                            while lane < 8 {
                                let (s, c) = branchless_sincos_q7(b[lane] as gcso_q7_t);
                                word_s[lane] += w * s;
                                word_c[lane] += w * c;
                                lane += 1;
                            }
                        }
                    } else {
                        all_byte_uniform = false;
                        all_word_uniform = false;
                        let mut cell = 0usize;
                        while cell < 8 {
                            let b = tw[cell].to_le_bytes();
                            let mut lane = 0usize;
                            while lane < 8 {
                                let (s, c) = branchless_sincos_q7(b[lane] as gcso_q7_t);
                                sum_s[cell][lane] += w * s;
                                sum_c[cell][lane] += w * c;
                                lane += 1;
                            }
                            cell += 1;
                        }
                    }
                }
            }
            if total_w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            if all_byte_uniform {
                let mag_sq = uniform_s * uniform_s + uniform_c * uniform_c;
                let has_mag = mag_sq > 1.0e-20;
                let q7 = branchless_select_q7(
                    has_mag,
                    branchless_atan2_to_q7_fast(uniform_s, uniform_c),
                    0,
                );
                let r_bar = branchless_select_f32(has_mag, 1.0, 0.0);
                let kappa_c = branchless_von_mises_kappa(r_bar);
                let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
                *profile_out = [branchless_select_q7(keep, q7, 0); MAX_ATTENTION_HEADS];
                return Ok((r_bar, kappa_c));
            } else if all_word_uniform {
                let mut s_cell = [0.0_f32; 8];
                let mut c_cell = [0.0_f32; 8];
                let mut lane = 0usize;
                while lane < 8 {
                    s_cell[lane] = word_s[lane] + uniform_s;
                    c_cell[lane] = word_c[lane] + uniform_c;
                    lane += 1;
                }
                let word0 = atan2_to_q7_word_x8_simd(&s_cell, &c_cell);
                candidate_words = [word0; 8];
                store_q7_words_64(&mut candidate, &candidate_words);
            } else {
                let has_word_or_uni = (uniform_s != 0.0)
                    | (uniform_c != 0.0)
                    | (word_s != [0.0_f32; 8])
                    | (word_c != [0.0_f32; 8]);
                if has_word_or_uni {
                    let mut cell = 0usize;
                    while cell < 8 {
                        let mut lane = 0usize;
                        while lane < 8 {
                            sum_s[cell][lane] += word_s[lane] + uniform_s;
                            sum_c[cell][lane] += word_c[lane] + uniform_c;
                            lane += 1;
                        }
                        cell += 1;
                    }
                }
                let mut cw = [0u64; 8];
                let mut cell = 0usize;
                while cell < 8 {
                    cw[cell] = atan2_to_q7_word_x8_simd(&sum_s[cell], &sum_c[cell]);
                    cell += 1;
                }
                candidate_words = cw;
                store_q7_words_64(&mut candidate, &candidate_words);
            }
        }

        let r_bar = circular_r_bar_from_words_64(&candidate_words);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
        if keep {
            *profile_out = candidate;
        } else {
            *profile_out = [0i8; MAX_ATTENTION_HEADS];
        }
        Ok((r_bar, kappa_c))
    }
}

impl PersonaPatcher for ResidualAdapterEngine {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
        ResidualAdapterEngine::apply_patch(self, patch_data)
    }
}
