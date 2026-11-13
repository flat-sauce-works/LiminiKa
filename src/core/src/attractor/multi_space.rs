// name: src/core/src/attractor/multi_space.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-domain dynamic 8-lane working space (MultiAttractorSpace).

use super::*;

// ===================================================================
// 6. Unified Phase Field & Dynamic 8-Lane Working Space
// ===================================================================

/// Unified Macro Attractor Space managing 512 total anchors (`8` hardware lanes $\times 64$ anchors)
/// on a single continuous phase field with automatic spillover and zero-hyperparameter proximity gravity.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiAttractorSpace {
    /// Relative lane weights (Cache Line 0: `offset 0..32`, initialized to neutral `1.0` and self-modulated by phase proximity).
    pub domain_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Active working lane bitmask (`offset 32..36`, `0xFF` = all 8 lanes active on the unified field).
    pub active_domains_mask: u32,
    /// Explicit 4-byte alignment padding before 64-bit counters (`offset 36..40`).
    pub reserved_flags: u32,
    /// Global token step counter across the unified phase space (`offset 40..48`).
    pub global_step_counter: u64,
    /// $8 \times 8$ cross-slot coupling bitmatrix (`u64`, `offset 48..56`).
    pub coupling_matrix_8x8: u64,
    /// Explicit padding completing Cache Line 0 (`offset 56..64`).
    pub reserved_padding: [u8; 8],
    /// 8 hardware-aligned working lanes (`offset 64..328768`, `8 * 64 = 512` unified anchors).
    pub fields: [AttractorFieldEngine; MAX_ATTRACTOR_DOMAINS],
}

// SAFETY: `MultiAttractorSpace` is `#[repr(C, align(64))]` where all fields are `Pod`.
unsafe impl Pod for MultiAttractorSpace {}

impl Default for MultiAttractorSpace {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAttractorSpace {
    /// Creates a new `MultiAttractorSpace` where all 8 lanes share full `MAX_ANCHORS_PER_FIELD = 64`
    /// capacity (`512` unified anchors) and neutral `1.0` baseline weights (zero manual hyperparameters).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        let cap = MAX_ANCHORS_PER_FIELD as u32;
        Self {
            fields: [
                AttractorFieldEngine::new(DOMAIN_SYSTEM_BASELINE, cap),
                AttractorFieldEngine::new(DOMAIN_PERSONA, cap),
                AttractorFieldEngine::new(DOMAIN_KNOWLEDGE, cap),
                AttractorFieldEngine::new(DOMAIN_PHASE_REPULSION, cap),
                AttractorFieldEngine::new(DOMAIN_USER_INTERACTION, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_5, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_6, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_7, cap),
            ],
            domain_weights: [1.0_f32; MAX_ATTRACTOR_DOMAINS],
            active_domains_mask: 0xFF,
            reserved_flags: 0,
            global_step_counter: 0,
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            reserved_padding: [0; 8],
        }
    }

    /// Resets all 8 domain lanes, weights, active masks, step counters, and coupling matrix in-place
    /// without allocating a temporary `320 KB` `MultiAttractorSpace` frame on the stack.
    #[inline]
    pub fn reset_all_lanes(&mut self) {
        let mut i = 0usize;
        while i < MAX_ATTRACTOR_DOMAINS {
            let field = &mut self.fields[i];
            let mut rem = field.active_count;
            if rem > 0 {
                let mut a = 0usize;
                while a < MAX_ANCHORS_PER_FIELD {
                    if field.anchors[a].is_active != 0 {
                        field.anchors[a].deactivate_in_place();
                        rem -= 1;
                        if rem == 0 {
                            break;
                        }
                    }
                    a += 1;
                }
            }
            field.active_count = 0;
            field.repulsion_gain = 0.0;
            field.repulsion_phase_deltas = [0; MAX_ATTENTION_HEADS];
            field.accumulator_pressure = 0.5;
            field.evaporation_rate = 0.01;
            field.dissipated_energy_total = 0.0;
            field.next_anchor_id = 1;
            field.step_counter = 0;
            self.domain_weights[i] = 1.0_f32;
            i += 1;
        }
        self.active_domains_mask = 0xFF;
        self.reserved_flags = 0;
        self.global_step_counter = 0;
        self.coupling_matrix_8x8 = DEFAULT_SLOT_COUPLING_MATRIX_8X8;
    }

    /// Returns a reference to the System Baseline working lane (Lane 0).
    #[inline(always)]
    #[must_use]
    pub const fn system_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_SYSTEM_BASELINE as usize]
    }

    /// Returns a mutable reference to the System Baseline working lane (Lane 0).
    #[inline(always)]
    pub fn system_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_SYSTEM_BASELINE as usize]
    }

    /// Returns a reference to the Persona working lane (Lane 1).
    #[inline(always)]
    #[must_use]
    pub const fn persona_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_PERSONA as usize]
    }

    /// Returns a mutable reference to the Persona working lane (Lane 1).
    #[inline(always)]
    pub fn persona_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_PERSONA as usize]
    }

    /// Returns a reference to the Knowledge working lane (Lane 2).
    #[inline(always)]
    #[must_use]
    pub const fn knowledge_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_KNOWLEDGE as usize]
    }

    /// Returns a mutable reference to the Knowledge working lane (Lane 2).
    #[inline(always)]
    pub fn knowledge_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_KNOWLEDGE as usize]
    }

    /// Returns a reference to the Phase Repulsion working lane (Lane 3).
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_PHASE_REPULSION as usize]
    }

    /// Returns a mutable reference to the Phase Repulsion working lane (Lane 3).
    #[inline(always)]
    pub fn repulsion_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_PHASE_REPULSION as usize]
    }

    /// Returns a reference to the User Interaction working lane (Lane 4).
    #[inline(always)]
    #[must_use]
    pub const fn user_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_USER_INTERACTION as usize]
    }

    /// Returns a mutable reference to the User Interaction working lane (Lane 4).
    #[inline(always)]
    pub fn user_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_USER_INTERACTION as usize]
    }

    /// Returns the total number of active anchors across the entire 512-anchor unified field.
    #[inline]
    #[must_use]
    pub fn total_unified_anchor_count(&self) -> u32 {
        let mut total = 0u32;
        for field in &self.fields {
            total = total.wrapping_add(field.active_count);
        }
        total
    }

    /// Selects the optimal target working lane (`0..=7`) for a new anchor on the unified field:
    /// uses `preferred_lane` if it has free capacity, or automatically spills over to the active
    /// non-repulsion lane with the lowest pressure / lowest displaceable anchor energy.
    #[inline]
    #[must_use]
    pub fn select_spillover_lane(&self, preferred_lane: u32) -> usize {
        let pref = (preferred_lane as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        if !self.fields[pref].is_saturated() {
            return pref;
        }
        let mut best_lane = pref;
        let mut lowest_pressure = f32::MAX;
        for (idx, field) in self.fields.iter().enumerate() {
            if idx == (DOMAIN_PHASE_REPULSION as usize) && pref != (DOMAIN_PHASE_REPULSION as usize)
            {
                continue;
            }
            let score = field.pressure_ratio() + field.min_displaceable_energy();
            if score < lowest_pressure {
                lowest_pressure = score;
                best_lane = idx;
            }
        }
        best_lane
    }

    /// Inserts an anchor into the unified 512-anchor phase field, automatically spilling over into
    /// adjacent working lanes if `preferred_lane` is saturated. Returns `(assigned_lane_id, anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `weight <= 0.0` or non-finite, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if all 512 slots resist displacement under pressure.
    #[inline]
    pub fn insert_unified_anchor(
        &mut self,
        preferred_lane: u32,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
        embedding: Option<&AlignedSlice32<'_, f32>>,
    ) -> GcsoResult<(u32, u32)> {
        let lane = self.select_spillover_lane(preferred_lane);
        self.active_domains_mask |= 1 << lane;
        let anchor_id = self.fields[lane].insert_anchor_internal(
            anchor_type,
            weight,
            embedding,
            None,
            false,
        )?;
        Ok((lane as u32, anchor_id))
    }

    /// Registers any [`EpisodicPhaseSample`] into the unified 512-anchor field using its
    /// `preferred_working_lane()` with automatic spillover. Returns `(assigned_lane_id, anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `sample.sample_mass() <= 0.0`, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if all 512 slots resist displacement.
    #[inline]
    pub fn register_episodic_sample<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        is_crystallized: bool,
    ) -> GcsoResult<(u32, u32)> {
        let lane = self.select_spillover_lane(sample.preferred_working_lane());
        self.active_domains_mask |= 1 << lane;
        let anchor_id = self.fields[lane].add_episodic_sample_anchor(sample, is_crystallized)?;
        Ok((lane as u32, anchor_id))
    }

    /// Autonomously updates `self.domain_weights` from each lane's actual phase resonance and
    /// potential energy relative to `trail` in division-free (`0 fdiv`) mode via
    /// [`branchless_inv_pos_f32`], eliminating manual hyperparameter tuning.
    #[inline]
    pub fn rebalance_weights_from_trail(
        &mut self,
        trail: &gcso_pointer_trail_t,
    ) -> [f32; MAX_ATTRACTOR_DOMAINS] {
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        let mut max_pull = 0.0_f32;
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);

        for (idx, field) in self.fields.iter().enumerate() {
            if ((self.active_domains_mask & (1 << idx)) != 0) & (field.active_count > 0) {
                let (_w, p) = field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
                pulls[idx] = p;
                max_pull = branchless_select_f32(p > max_pull, p, max_pull);
            }
        }

        let has_max = max_pull > 1.0e-6;
        let inv_max = branchless_inv_pos_f32(branchless_select_f32(has_max, max_pull, 1.0));
        for (idx, &p) in pulls.iter().enumerate() {
            let norm_w = branchless_clamp_f32(0.25 + 1.75 * (p * inv_max), 0.25, 2.0);
            self.domain_weights[idx] = branchless_select_f32(has_max & (p > 0.0), norm_w, 1.0);
        }
        self.domain_weights
    }

    /// Projects the unified 512-anchor phase field around `trail` into the 8 hardware SIMD working
    /// lanes (`bank_out: &mut [[gcso_q7_t; 64]; 8]`) and outputs each lane's resonant pull force
    /// (`lane_pulls_out: &mut [f32; 8]`), applying mutual inhibition from Lane 3 (`Repulsion`).
    #[inline]
    pub fn project_active_field_bank_8(
        &self,
        trail: &gcso_pointer_trail_t,
        bank_out: &mut [[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        lane_pulls_out: &mut [f32; MAX_ATTRACTOR_DOMAINS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);

        let rep_field = &self.fields[DOMAIN_PHASE_REPULSION as usize];
        let rep_w_gain = branchless_relu_f32(self.domain_weights[DOMAIN_PHASE_REPULSION as usize]);
        let (rep_words, rep_pull) = if ((rep_field.active_count > 0)
            | (rep_field.repulsion_gain > 0.0))
            & (rep_w_gain > 0.0)
        {
            let (rw, rp) = rep_field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            (rw, rp * rep_w_gain)
        } else {
            ([0u64; 8], 0.0_f32)
        };
        let mut total_pull = 0.0_f32;
        for (idx, field) in self.fields.iter().enumerate() {
            let enabled = (self.active_domains_mask & (1 << idx)) != 0;
            let w = branchless_relu_f32(self.domain_weights[idx]);
            if (!enabled) | (field.active_count == 0) | (w <= 0.0) {
                bank_out[idx] = [0i8; MAX_ATTENTION_HEADS];
                lane_pulls_out[idx] = 0.0;
                continue;
            }
            let (mut slot_w, raw_pull) =
                field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            let mut pull = raw_pull * w;
            if (idx != (DOMAIN_PHASE_REPULSION as usize)) & (rep_pull > 1.0e-6) {
                pull = branchless_q7_inhibit_words_64(&mut slot_w, pull, &rep_words, 0.5);
            }
            store_q7_words_64(&mut bank_out[idx], &slot_w);
            lane_pulls_out[idx] = pull;
            total_pull += pull;
        }
        branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        )
    }

    /// Advances global step counter across all working lanes and evaluates evaporative decay
    /// (touching only Cache Line 0 of empty lanes and running anchor decay only when `active_count > 0`).
    #[inline]
    pub fn advance_step(&mut self) {
        let next_step = self.global_step_counter.wrapping_add(1);
        self.global_step_counter = next_step;
        for field in &mut self.fields {
            field.step_counter = next_step;
            if field.active_count != 0 {
                field.apply_evaporative_decay();
            }
        }
    }

    /// Sets relative lane weighting scalar (`weight >= 0.0`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `weight` is negative/non-finite.
    #[inline]
    pub fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(weight)
            || weight < 0.0
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.domain_weights[domain_id as usize] = weight;
        Ok(())
    }

    /// Enables or disables a specific working lane (`0..=7`) via bitmask.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bit = 1u32 << idx;
        let mask = (enabled as u32).wrapping_neg();
        self.active_domains_mask = (self.active_domains_mask & !bit) | (bit & mask);
        Ok(())
    }

    /// Retrieves a mutable reference to a specific working lane by ID (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn domain_mut(&mut self, domain_id: u32) -> GcsoResult<&mut AttractorFieldEngine> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(&mut self.fields[idx])
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Retrieves an immutable reference to a specific working lane by ID (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn domain(&self, domain_id: u32) -> GcsoResult<&AttractorFieldEngine> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(&self.fields[idx])
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Exports accumulator pressure metrics for a specific working lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn get_accumulator_summary(&self, domain_id: u32) -> GcsoResult<DomainAccumulatorSummary> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(self.fields[idx].accumulator_summary(self.domain_weights[idx]))
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Exports accumulator pressure summaries across all 8 working lanes.
    #[inline]
    #[must_use]
    pub fn get_all_accumulator_summaries(
        &self,
    ) -> [DomainAccumulatorSummary; MAX_ATTRACTOR_DOMAINS] {
        let mut summaries = [DomainAccumulatorSummary::default(); MAX_ATTRACTOR_DOMAINS];
        for (idx, field) in self.fields.iter().enumerate() {
            summaries[idx] = field.accumulator_summary(self.domain_weights[idx]);
        }
        summaries
    }

    /// Identifies the dominant working lane (`0..=7`) currently holding maximum potential energy.
    #[inline]
    #[must_use]
    pub fn find_dominant_domain(&self) -> (u32, f32) {
        let mut max_energy = -1.0_f32;
        let mut dominant_id = DOMAIN_SYSTEM_BASELINE;

        for (idx, field) in self.fields.iter().enumerate() {
            if ((self.active_domains_mask & (1 << idx)) == 0) | (field.active_count == 0) {
                continue;
            }
            let weight = branchless_relu_f32(self.domain_weights[idx]);
            if weight <= 0.0 {
                continue;
            }
            let weighted_energy = field.total_potential_energy() * weight;
            if weighted_energy > max_energy {
                max_energy = weighted_energy;
                dominant_id = idx as u32;
            }
        }
        if max_energy < 0.0 {
            (DOMAIN_SYSTEM_BASELINE, 0.0)
        } else {
            (dominant_id, max_energy)
        }
    }

    /// Register-resident core evaluator for unified multi-domain superposition returning
    /// `(composite_words_64, pull_force)` directly in 8 `u64` SWAR registers.
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_multi_domain_superposition_words(
        &self,
        init_w: &[u64; 8],
        init_nonzero: bool,
    ) -> ([u64; 8], f32) {
        let rep_field = &self.fields[DOMAIN_PHASE_REPULSION as usize];
        let rep_w_gain = branchless_relu_f32(self.domain_weights[DOMAIN_PHASE_REPULSION as usize]);
        let (rep_words, rep_pull) = if ((rep_field.active_count > 0)
            | (rep_field.repulsion_gain > 0.0))
            & (rep_w_gain > 0.0)
        {
            let (rw, rp) = rep_field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            (rw, rp * rep_w_gain)
        } else {
            ([0u64; 8], 0.0_f32)
        };

        let mut running_w = [0u64; 8];
        let mut running_pull = 0.0_f32;
        let mut primary_slot = 0u32;

        for (idx, field) in self.fields.iter().enumerate() {
            if idx == (DOMAIN_PHASE_REPULSION as usize) {
                continue;
            }
            let enabled = (self.active_domains_mask & (1 << idx)) != 0;
            let w = branchless_relu_f32(self.domain_weights[idx]);
            if (!enabled) | (field.active_count == 0) | (w <= 0.0) {
                continue;
            }
            let (mut slot_w, raw_pull) =
                field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            let mut pull = raw_pull * w;
            if rep_pull > 1.0e-6 {
                pull = branchless_q7_inhibit_words_64(&mut slot_w, pull, &rep_words, 0.5);
            }
            if pull <= 0.0 {
                continue;
            }
            if running_pull <= 0.0 {
                running_w = slot_w;
                running_pull = pull;
                primary_slot = idx as u32;
            } else {
                let coupled =
                    is_slot_pair_coupled(self.coupling_matrix_8x8, primary_slot, idx as u32);
                let syn_gain = branchless_select_f32(coupled, 0.35, 0.0);
                let (fused_w, new_pull, _coh) = branchless_q7_synergy_fuse_words_64(
                    &running_w,
                    running_pull,
                    &slot_w,
                    pull,
                    syn_gain,
                );
                running_w = fused_w;
                running_pull = new_pull;
            }
        }

        if running_pull <= 0.0 {
            return (*init_w, 0.0);
        }

        let pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (running_w, pull)
    }

    /// Evaluates unified field superposition of gravitational pulls and non-linear cross-slot
    /// phase resonance (`branchless_q7_synergy_fuse_64`) in zero-allocation mode.
    #[inline]
    pub fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);
        let (running_w, pull) =
            self.evaluate_multi_domain_superposition_words(init_w, init_nonzero);
        store_q7_words_64(composite_phase_out, &running_w);
        pull
    }

    /// Exports the grown persona profile, circular concentration $\kappa_c$, lane weights, and up to
    /// `MAX_ANCHORS_PER_FIELD` (`64`) crystallized/active sample points across the unified space into
    /// a compact 8,448-byte [`AttractorGrowthSnapshot`] (`Pod`) stamped with an IEEE 802.3 CRC32 checksum.
    #[inline]
    pub fn export_growth_snapshot(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        snapshot_out: &mut AttractorGrowthSnapshot,
    ) -> u32 {
        let init_w = trail_phase_words_ref(reference_trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);
        let (run_w, _pull) = self.evaluate_multi_domain_superposition_words(init_w, init_nonzero);
        let rep_field = &self.fields[DOMAIN_PHASE_REPULSION as usize];
        let prev_clear_end = snapshot_out.init_header_and_persona_from_words(
            self.global_step_counter,
            self.coupling_matrix_8x8,
            self.active_domains_mask,
            &self.domain_weights,
            rep_field.repulsion_gain,
            &rep_field.repulsion_phase_deltas,
            self.total_potential_energy(),
            &run_w,
        );

        let mut out_idx = 0usize;
        let mut cryst_cnt = 0u32;
        let mut transient_coords = [(0u8, 0u8); MAX_ANCHORS_PER_FIELD];
        let mut transient_len = 0usize;

        // Single pass across active fields (`field.active_count > 0`), stopping as soon as all
        // active anchors in each lane are visited (`0` scans over empty slots).
        for (f_idx, field) in self.fields.iter().enumerate() {
            let mut rem = field.active_count;
            if rem == 0 {
                continue;
            }
            let cap = (field.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
            for (a_idx, anchor) in field.anchors[..cap].iter().enumerate() {
                if rem == 0 {
                    break;
                }
                if anchor.is_active != 0 {
                    rem -= 1;
                    if anchor.is_crystallized != 0 {
                        if out_idx < MAX_ANCHORS_PER_FIELD {
                            snapshot_out.anchors[out_idx] =
                                CrystallizedAnchorRecord::from_anchor(anchor);
                            cryst_cnt = cryst_cnt.wrapping_add(1);
                            out_idx += 1;
                        }
                    } else if transient_len < MAX_ANCHORS_PER_FIELD {
                        transient_coords[transient_len] = (f_idx as u8, a_idx as u8);
                        transient_len += 1;
                    }
                }
            }
        }
        let mut t_i = 0usize;
        while t_i < transient_len && out_idx < MAX_ANCHORS_PER_FIELD {
            let (f_idx, a_idx) = transient_coords[t_i];
            let anchor = &self.fields[f_idx as usize].anchors[a_idx as usize];
            snapshot_out.anchors[out_idx] = CrystallizedAnchorRecord::from_anchor(anchor);
            out_idx += 1;
            t_i += 1;
        }
        snapshot_out.finalize_anchors_and_crc32(out_idx, cryst_cnt, prev_clear_end)
    }

    /// Restores the unified `MultiAttractorSpace` from a verified [`AttractorGrowthSnapshot`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`GCSO_ERROR_VERSION_MISMATCH`] if
    /// `snapshot.verify_integrity()` fails.
    #[inline]
    pub fn restore_from_growth_snapshot(
        &mut self,
        snapshot: &AttractorGrowthSnapshot,
    ) -> GcsoResult<()> {
        snapshot.verify_integrity()?;
        self.clear_all_domains()?;
        self.global_step_counter = snapshot.global_step_counter;
        self.coupling_matrix_8x8 = snapshot.coupling_matrix_8x8;
        self.active_domains_mask = snapshot.active_domains_mask;
        self.domain_weights = snapshot.domain_weights;
        for field in &mut self.fields {
            field.step_counter = snapshot.global_step_counter;
        }
        self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_gain = snapshot.repulsion_gain;
        self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_phase_deltas =
            snapshot.repulsion_phase_deltas;

        let count = (snapshot.active_anchor_count as usize).min(MAX_ANCHORS_PER_FIELD);
        for rec in snapshot.anchors.iter().take(count) {
            if rec.is_active == 0 {
                continue;
            }
            let lane = self.select_spillover_lane(rec.domain_id);
            let field = &mut self.fields[lane];
            let slot_idx = field.allocate_anchor_slot(rec.weight.max(0.1))?;
            rec.write_to_anchor_in_place(&mut field.anchors[slot_idx], Some(lane as u32));
            if rec.id >= field.next_anchor_id {
                field.next_anchor_id = rec.id.wrapping_add(1).max(1);
            }
        }
        Ok(())
    }

    /// Serializes a compact 8,448-byte [`AttractorGrowthSnapshot`] directly into `dst_bytes` at `offset`.
    ///
    /// Returns the number of bytes written (`size_of::<AttractorGrowthSnapshot>() = 8448`).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < offset + 8448`.
    #[inline]
    pub fn serialize_growth_snapshot_to_bytes(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        dst_bytes: &mut [u8],
        offset: usize,
    ) -> GcsoResult<usize> {
        if let Ok(snap_mut) = AttractorGrowthSnapshot::from_bytes_mut_at(dst_bytes, offset) {
            let _ = self.export_growth_snapshot(reference_trail, snap_mut);
            return Ok(size_of::<AttractorGrowthSnapshot>());
        }
        let mut snap = AttractorGrowthSnapshot::new();
        let _ = self.export_growth_snapshot(reference_trail, &mut snap);
        snap.write_unaligned_to(dst_bytes, offset)?;
        Ok(size_of::<AttractorGrowthSnapshot>())
    }

    /// Zero-copy casts, verifies CRC32, and restores `self` from a 64-byte-aligned `.gcso` ZIMMS
    /// byte buffer at `offset`, returning the zero-copy `&'a AttractorGrowthSnapshot` reference.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`], [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`], or [`GCSO_ERROR_VERSION_MISMATCH`] on failure.
    #[inline]
    pub fn restore_from_snapshot_bytes_zero_copy<'a>(
        &mut self,
        src_bytes: &'a [u8],
        offset: usize,
    ) -> GcsoResult<&'a AttractorGrowthSnapshot> {
        let snap_ref = AttractorGrowthSnapshot::from_bytes_verified_at(src_bytes, offset)?;
        self.restore_from_growth_snapshot(snap_ref)?;
        Ok(snap_ref)
    }

    /// Restores `self` from a potentially unaligned `.gcso` byte slice at `offset` after verifying
    /// magic, ABI version, and IEEE 802.3 CRC32 integrity.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_CONTAINER_CORRUPTED`],
    /// or [`GCSO_ERROR_VERSION_MISMATCH`] on failure.
    #[inline]
    pub fn restore_from_snapshot_bytes_unaligned(
        &mut self,
        src_bytes: &[u8],
        offset: usize,
    ) -> GcsoResult<AttractorGrowthSnapshot> {
        if let Ok(snap_ref) = AttractorGrowthSnapshot::from_bytes_verified_at(src_bytes, offset) {
            self.restore_from_growth_snapshot(snap_ref)?;
            return Ok(*snap_ref);
        }
        let snap = AttractorGrowthSnapshot::read_unaligned_from(src_bytes, offset)?;
        self.restore_from_growth_snapshot(&snap)?;
        Ok(snap)
    }

    /// Adds a system prompt anchor to `domain_id`, spilling over to another lane if full.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `prompt` is empty.
    #[inline]
    pub fn add_system_prompt_to_domain(
        &mut self,
        domain_id: u32,
        prompt: &str,
        weight: f32,
    ) -> GcsoResult<u32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS || prompt.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane = self.select_spillover_lane(domain_id);
        self.active_domains_mask |= 1 << lane;
        self.fields[lane].add_system_prompt_anchor(prompt, weight)
    }

    /// Adds an embedding anchor to `domain_id`, spilling over to another lane if full.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `embedding` is empty.
    #[inline]
    pub fn add_embedding_to_domain(
        &mut self,
        domain_id: u32,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS || embedding.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (_lane, id) =
            self.insert_unified_anchor(domain_id, anchor_type, weight, Some(embedding))?;
        Ok(id)
    }

    /// Registers a knowledge source embedding into the unified field (preferring Lane 2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `embedding` is empty or `source_weight <= 0.0`.
    #[inline]
    pub fn register_knowledge_source_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        source_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_KNOWLEDGE,
            embedding,
            GCSO_ANCHOR_TYPE_TOPOLOGICAL,
            source_weight,
        )
    }

    /// Registers a persona modulation patch into the unified field (preferring Lane 1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `persona_vector` is empty or `persona_weight <= 0.0`.
    #[inline]
    pub fn register_persona_patch_anchor(
        &mut self,
        persona_vector: &AlignedSlice32<'_, f32>,
        persona_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_PERSONA,
            persona_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            persona_weight,
        )
    }

    /// Registers a user interaction context vector into the unified field (preferring Lane 4).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `context_vector` is empty or `context_weight <= 0.0`.
    #[inline]
    pub fn register_user_context_anchor(
        &mut self,
        context_vector: &AlignedSlice32<'_, f32>,
        context_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_USER_INTERACTION,
            context_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            context_weight,
        )
    }

    /// Clears all anchors across all 8 working lanes in the unified space.
    ///
    /// # Errors
    /// Propagates any error from [`AttractorFieldEngine::clear_anchors`].
    #[inline]
    pub fn clear_all_domains(&mut self) -> GcsoResult<()> {
        for field in &mut self.fields {
            field.clear_anchors()?;
        }
        Ok(())
    }
}

impl AttractorField for MultiAttractorSpace {
    #[inline(always)]
    fn anchor_count(&self) -> u32 {
        self.total_unified_anchor_count()
    }

    #[inline(always)]
    fn apply_evaporative_decay(&mut self) {
        self.advance_step();
    }

    #[inline]
    fn total_potential_energy(&self) -> f32 {
        let mut total = 0.0_f32;
        for (idx, field) in self.fields.iter().enumerate() {
            if (self.active_domains_mask & (1 << idx)) != 0 {
                total +=
                    field.total_potential_energy() * branchless_relu_f32(self.domain_weights[idx]);
            }
        }
        branchless_flush_subnormal_f32(total)
    }

    #[inline]
    fn pressure_ratio(&self) -> f32 {
        let total_cap = MAX_ATTRACTOR_DOMAINS * MAX_ANCHORS_PER_FIELD;
        branchless_flush_subnormal_f32(
            (self.total_unified_anchor_count() as f32) * fast_inv_len_f32(total_cap),
        )
    }

    #[inline]
    fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
        let mut weakest_lane = None;
        let mut lowest_e = f32::MAX;
        for (idx, field) in self.fields.iter().enumerate() {
            if field.active_count > 0 {
                let e = field.min_displaceable_energy();
                if e < lowest_e {
                    lowest_e = e;
                    weakest_lane = Some(idx);
                }
            }
        }
        self.fields[weakest_lane?].evaporate_weakest_anchor()
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32> {
        self.add_system_prompt_to_domain(DOMAIN_PERSONA, prompt, weight)
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        let pref = if anchor_type == GCSO_ANCHOR_TYPE_PHASE_REPULSE {
            DOMAIN_PHASE_REPULSION
        } else if anchor_type == GCSO_ANCHOR_TYPE_TOPOLOGICAL {
            DOMAIN_KNOWLEDGE
        } else {
            DOMAIN_PERSONA
        };
        let (_lane, id) = self.insert_unified_anchor(pref, anchor_type, weight, Some(embedding))?;
        Ok(id)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32> {
        if dim == 0 || vec.len() < dim {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.add_embedding_anchor(vec, anchor_type, 1.0)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()> {
        self.fields[DOMAIN_PHASE_REPULSION as usize].inject_phase_repulsion(repulsion_deltas, gain)
    }

    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        Ok(self.evaluate_multi_domain_superposition(trail, composite_phase_out))
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
        let mut total = 0u32;
        for field in &mut self.fields {
            total = total.wrapping_add(field.aggregate_bottom_up()?);
        }
        Ok(total)
    }

    #[inline]
    fn clear_anchors(&mut self) -> GcsoResult<()> {
        self.clear_all_domains()
    }
}

impl MultiDomainAttractorSpace for MultiAttractorSpace {
    #[inline(always)]
    fn advance_step(&mut self) {
        Self::advance_step(self);
    }

    #[inline(always)]
    fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()> {
        Self::set_domain_weight(self, domain_id, weight)
    }

    #[inline(always)]
    fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()> {
        Self::set_domain_enabled(self, domain_id, enabled)
    }

    #[inline(always)]
    fn find_dominant_domain(&self) -> (u32, f32) {
        Self::find_dominant_domain(self)
    }

    #[inline(always)]
    fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        Self::evaluate_multi_domain_superposition(self, trail, composite_phase_out)
    }

    #[inline(always)]
    fn clear_all_domains(&mut self) -> GcsoResult<()> {
        Self::clear_all_domains(self)
    }

    #[inline(always)]
    fn slot_coupling_matrix(&self) -> u64 {
        self.coupling_matrix_8x8
    }

    #[inline]
    fn evaluate_slot_force(
        &self,
        domain_id: u32,
        trail: &gcso_pointer_trail_t,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if ((self.active_domains_mask & (1 << idx)) == 0) | (self.fields[idx].active_count == 0) {
            *slot_phase_out = [0i8; MAX_ATTENTION_HEADS];
            return Ok(0.0);
        }
        let pull = self.fields[idx].evaluate_composite_attractor_force(trail, slot_phase_out);
        Ok(pull * branchless_relu_f32(self.domain_weights[idx]))
    }
}

impl PersonaPatcher for MultiAttractorSpace {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
        if patch_data.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut phases = [0i8; MAX_ATTENTION_HEADS];
        let len = patch_data.len();
        if len >= MAX_ATTENTION_HEADS {
            // SAFETY: `patch_data.len() >= 64` and `[u8; 64]` has identical size and alignment (1) as `[gcso_q7_t; 64]`.
            let patch_ref = unsafe {
                &*patch_data
                    .as_ptr()
                    .cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()
            };
            phases = *patch_ref;
        } else {
            for (h, slot) in phases.iter_mut().enumerate() {
                *slot = patch_data[h % len] as gcso_q7_t;
            }
        }
        let lane = self.select_spillover_lane(DOMAIN_PERSONA);
        self.active_domains_mask |= 1 << lane;
        let _ = self.fields[lane].insert_anchor_internal(
            GCSO_ANCHOR_TYPE_EMBEDDING,
            1.5,
            None,
            Some(&phases),
            false,
        )?;
        Ok(())
    }
}

const _: () = {
    assert!(align_of::<AttractorGrowthReport>() == 16);
    assert!(size_of::<AttractorGrowthReport>() == 16);
    assert!(align_of::<CrystallizedAnchorRecord>() == 64);
    assert!(size_of::<CrystallizedAnchorRecord>() == 128);
    assert!(offset_of!(CrystallizedAnchorRecord, last_active_step) == 104);
    assert!(align_of::<AttractorGrowthSnapshot>() == 64);
    assert!(size_of::<AttractorGrowthSnapshot>() == 8448);
    assert!(offset_of!(AttractorGrowthSnapshot, persona_phase_profile) == 64);
    assert!(offset_of!(AttractorGrowthSnapshot, anchors) == 256);
    assert!(align_of::<DomainAccumulatorSummary>() == 16);
    assert!(size_of::<DomainAccumulatorSummary>() == 32);
    assert!(align_of::<AttractorAnchor>() == 64);
    assert!(size_of::<AttractorAnchor>() == 640);
    assert!(offset_of!(AttractorAnchor, phase_deltas) == 64);
    assert!(offset_of!(AttractorAnchor, vector_data) == 128);
    assert!(align_of::<AttractorFieldEngine>() == 64);
    assert!(offset_of!(AttractorFieldEngine, repulsion_phase_deltas) == 64);
    assert!(offset_of!(AttractorFieldEngine, anchors) == 128);
    assert!(align_of::<MultiAttractorSpace>() == 64);
    assert!(offset_of!(MultiAttractorSpace, fields) == 64);
};
