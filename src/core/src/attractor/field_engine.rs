// name: src/core/src/attractor/field_engine.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-field unified phase-manifold attractor engine (AttractorFieldEngine).

use super::*;

// ===================================================================
// 5. Single-Lane Attractor Field Accumulator Engine
// ===================================================================

/// Dynamic Attractor Field Engine with Accumulator Pressure Dynamics and $S^1$ Geodesic Phase-Proximity Gravity.
///
/// # Cache-Line Layout (`#[repr(C, align(64))]`)
/// - **Cache Line 0 (`offset 0..64`)**: Hot lane control metadata (`active_count`, `capacity`,
///   `domain_id`, `repulsion_gain`, `accumulator_pressure`, `evaporation_rate`,
///   `dissipated_energy_total`, `next_anchor_id`, `step_counter`, `reserved_padding`).
/// - **Cache Line 1 (`offset 64..128`)**: `repulsion_phase_deltas: [gcso_q7_t; 64]`, strictly
///   64-byte cacheline-aligned (`0` cacheline splits).
/// - **Cache Lines 2..641 (`offset 128..41088`)**: `anchors: [AttractorAnchor; 64]`, each strictly
///   64-byte cacheline-aligned (`640` bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorFieldEngine {
    /// Number of active anchors registered in this lane.
    pub active_count: u32,
    /// Maximum capacity of this lane (`<= 64`).
    pub capacity: u32,
    /// Target working lane identifier (`0..=7`).
    pub domain_id: u32,
    /// Global phase repulsion gain factor.
    pub repulsion_gain: f32,
    /// Accumulator pressure factor controlling minimum threshold for evaporative displacement.
    pub accumulator_pressure: f32,
    /// Evaporation rate per step applied to stigmergic density under external pressure.
    pub evaporation_rate: f32,
    /// Cumulative energy dissipated through anchor evaporation or displacement.
    pub dissipated_energy_total: f32,
    /// Auto-incrementing anchor ID generator.
    pub(super) next_anchor_id: u32,
    /// Internal token step counter for LRU and temporal decay tracking.
    pub step_counter: u64,
    /// Explicit padding completing Cache Line 0 (`offset 40..64`).
    pub reserved_padding: [u8; 24],
    /// Global anti-phase repulsion profile vector ($-\boldsymbol{\Delta\theta}$, Cache Line 1: `offset 64..128`).
    pub repulsion_phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Fixed-capacity storage array for topological anchors (`64` slots, `offset 128..41088`).
    pub anchors: [AttractorAnchor; MAX_ANCHORS_PER_FIELD],
}

// SAFETY: `AttractorFieldEngine` is `#[repr(C, align(64))]` where all fields are `Pod`.
unsafe impl Pod for AttractorFieldEngine {}

impl Default for AttractorFieldEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(DOMAIN_SYSTEM_BASELINE, MAX_ANCHORS_PER_FIELD as u32)
    }
}

impl AttractorFieldEngine {
    /// Borrows `self.repulsion_phase_deltas` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorFieldEngine`], guaranteed 64-byte cacheline-aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorFieldEngine` is `#[repr(C, align(64))]` and `repulsion_phase_deltas`
        // is at byte offset 64 (`64 % 64 == 0`, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .repulsion_phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Creates a new `AttractorFieldEngine` instance for a specific working lane.
    #[inline]
    #[must_use]
    pub const fn new(domain_id: u32, capacity: u32) -> Self {
        let cap = branchless_select_u32(
            capacity > (MAX_ANCHORS_PER_FIELD as u32),
            MAX_ANCHORS_PER_FIELD as u32,
            capacity,
        );
        Self {
            active_count: 0,
            capacity: cap,
            domain_id,
            repulsion_gain: 0.0,
            accumulator_pressure: 0.5,
            evaporation_rate: 0.01,
            dissipated_energy_total: 0.0,
            next_anchor_id: 1,
            step_counter: 0,
            reserved_padding: [0; 24],
            repulsion_phase_deltas: [0; MAX_ATTENTION_HEADS],
            anchors: [AttractorAnchor {
                id: 0,
                anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
                weight: 1.0,
                pull_force: 1.0,
                stigmergic_density: 1.0,
                half_life_steps: 1000,
                last_active_step: 0,
                domain_id,
                vector_dim: 0,
                is_active: 0,
                is_crystallized: 0,
                reserved_padding: [0; 24],
                phase_deltas: [0; MAX_ATTENTION_HEADS],
                vector_data: [0.0; MAX_EMBEDDING_DIM],
            }; MAX_ANCHORS_PER_FIELD],
        }
    }

    /// Returns the active anchor count.
    #[inline(always)]
    #[must_use]
    pub const fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Returns the working lane / domain identifier.
    #[inline(always)]
    #[must_use]
    pub const fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Returns `true` if this lane is saturated (`active_count >= capacity`).
    #[inline(always)]
    #[must_use]
    pub const fn is_saturated(&self) -> bool {
        self.active_count >= self.capacity
    }

    /// Returns the number of crystallized anchors in this lane.
    #[inline]
    #[must_use]
    pub fn crystallized_count(&self) -> u32 {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return 0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut count = 0u32;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active != 0 {
                count = count.wrapping_add(u32::from(anchor.is_crystallized != 0));
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        count
    }

    /// Calculates the total potential energy currently stored in this lane.
    #[inline]
    #[must_use]
    pub fn total_potential_energy(&self) -> f32 {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return 0.0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut total = 0.0_f32;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active != 0 {
                total += anchor.compute_energy_score(self.step_counter);
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        branchless_flush_subnormal_f32(total)
    }

    /// Returns the lowest active non-crystallized anchor energy in this lane (or `0.0` if unsaturated).
    #[inline]
    #[must_use]
    pub fn min_displaceable_energy(&self) -> f32 {
        if !self.is_saturated() {
            return 0.0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut lowest = f32::MAX;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active == 0 {
                return 0.0;
            }
            if anchor.is_crystallized == 0 {
                let e = anchor.compute_energy_score(self.step_counter);
                if e < lowest {
                    lowest = e;
                }
            }
        }
        lowest
    }

    /// Calculates the current pressure ratio (`active_count / capacity`) in division-free (`0 fdiv`) mode.
    #[inline(always)]
    #[must_use]
    pub const fn pressure_ratio(&self) -> f32 {
        let cap = self.capacity as usize;
        branchless_flush_subnormal_f32((self.active_count as f32) * fast_inv_len_f32(cap))
    }

    /// Exports accumulator pressure metrics summary for this lane.
    #[inline]
    #[must_use]
    pub fn accumulator_summary(&self, domain_weight: f32) -> DomainAccumulatorSummary {
        DomainAccumulatorSummary {
            domain_id: self.domain_id,
            active_count: self.active_count,
            capacity: self.capacity,
            crystallized_count: self.crystallized_count(),
            pressure_ratio: self.pressure_ratio(),
            total_potential_energy: self.total_potential_energy(),
            dissipated_energy_total: self.dissipated_energy_total,
            domain_weight,
        }
    }

    /// Evaluates evaporative pressure decay across all active non-crystallized anchors in-place.
    #[inline]
    pub fn apply_evaporative_decay(&mut self) {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let retain = branchless_clamp_f32(1.0 - self.evaporation_rate, 0.0, 1.0);
        let min_energy = 0.05 * branchless_relu_f32(self.accumulator_pressure);
        let mut deactivated_count = 0u32;

        for anchor in &mut self.anchors[..cap] {
            if anchor.is_active == 0 {
                continue;
            }
            if anchor.is_crystallized == 0 {
                anchor.stigmergic_density =
                    branchless_flush_subnormal_f32(anchor.stigmergic_density * retain);
                let current_energy = anchor.compute_energy_score(self.step_counter);
                if current_energy < min_energy {
                    self.dissipated_energy_total += current_energy;
                    anchor.deactivate_in_place();
                    deactivated_count = deactivated_count.wrapping_add(1);
                }
            }
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        self.active_count = self.active_count.saturating_sub(deactivated_count);
    }

    /// Finds the weakest non-crystallized anchor slot (or oldest LRU slot if all are crystallized).
    #[inline]
    fn displace_weakest_anchor_slot(&self) -> Option<usize> {
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut lowest_score = f32::MAX;
        let mut target_idx = None;
        let mut oldest_step = u64::MAX;
        let mut lru_idx = None;

        for (idx, anchor) in self.anchors[..cap].iter().enumerate() {
            if anchor.is_active == 0 {
                return Some(idx);
            }
            if anchor.is_crystallized == 0 {
                let score = anchor.compute_energy_score(self.step_counter);
                if score < lowest_score {
                    lowest_score = score;
                    target_idx = Some(idx);
                }
            }
            if anchor.last_active_step < oldest_step {
                oldest_step = anchor.last_active_step;
                lru_idx = Some(idx);
            }
        }
        target_idx.or(lru_idx)
    }

    /// Allocates an anchor slot enforcing Accumulator Pressure Dynamics (Evaporative Displacement).
    #[inline]
    pub(super) fn allocate_anchor_slot(&mut self, incoming_energy: f32) -> GcsoResult<usize> {
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        if !self.is_saturated() {
            for (idx, anchor) in self.anchors[..cap].iter().enumerate() {
                if anchor.is_active == 0 {
                    self.active_count = self.active_count.wrapping_add(1);
                    return Ok(idx);
                }
            }
        }

        let slot_idx = self
            .displace_weakest_anchor_slot()
            .ok_or(GCSO_ERROR_ACTION_HUB_FULL)?;
        let target_anchor = &self.anchors[slot_idx];
        if target_anchor.is_active != 0 {
            let existing_score = target_anchor.compute_energy_score(self.step_counter);
            if incoming_energy < existing_score * self.accumulator_pressure {
                return Err(GCSO_ERROR_ACTION_HUB_FULL);
            }
            self.dissipated_energy_total += existing_score;
        }
        Ok(slot_idx)
    }

    /// Internal helper registering an anchor with optional embedding or deterministic prompt hash phases.
    #[inline]
    pub(super) fn insert_anchor_internal(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
        embedding: Option<&AlignedSlice32<'_, f32>>,
        prompt_phases: Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]>,
        is_crystallized: bool,
    ) -> GcsoResult<u32> {
        if !branchless_is_finite_f32(weight) || weight <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let slot_idx = self.allocate_anchor_slot(weight)?;
        let anchor_id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.wrapping_add(1);
        if self.next_anchor_id == 0 {
            self.next_anchor_id = 1;
        }

        let step = self.step_counter;
        let dom_id = self.domain_id;
        let dst = &mut self.anchors[slot_idx];
        dst.id = anchor_id;
        dst.anchor_type = anchor_type;
        dst.weight = weight;
        dst.pull_force = 1.0;
        dst.stigmergic_density = 1.0;
        dst.half_life_steps = 1000;
        dst.last_active_step = step;
        dst.domain_id = dom_id;
        dst.vector_dim = 0;
        dst.is_active = 1;
        dst.is_crystallized = u8::from(is_crystallized);

        let mut phases = match prompt_phases {
            Some(p_phases) => *p_phases,
            None => [0i8; MAX_ATTENTION_HEADS],
        };
        if let Some(emb) = embedding {
            let copy_len = emb.len().min(MAX_EMBEDDING_DIM);
            if copy_len > 0 {
                let src_slice = &emb.as_slice()[..copy_len];
                dst.vector_data[..copy_len].copy_from_slice(src_slice);
                dst.vector_dim = copy_len as u16;
                if prompt_phases.is_none() {
                    phases = project_f32_slice_to_q7_phases(src_slice);
                }
            }
        }
        if anchor_type == GCSO_ANCHOR_TYPE_PHASE_REPULSE {
            phases = branchless_q7_conjugate_64(&phases);
        }
        dst.phase_deltas = phases;
        Ok(anchor_id)
    }

    /// Directly registers any [`EpisodicPhaseSample`] as an anchor in this lane.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if sample mass is non-positive, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if displacement fails under accumulator pressure.
    #[inline]
    pub fn add_episodic_sample_anchor<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        is_crystallized: bool,
    ) -> GcsoResult<u32> {
        let phases = sample.sample_phase_coords(self.domain_id);
        let mass = sample.sample_mass();
        self.insert_anchor_internal(
            sample.sample_anchor_type(),
            mass,
            None,
            Some(&phases),
            is_crystallized,
        )
    }

    /// Finds an immutable anchor by unique anchor ID in this lane.
    #[inline]
    #[must_use]
    pub fn get_anchor(&self, anchor_id: u32) -> Option<&AttractorAnchor> {
        if anchor_id == 0 {
            return None;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        self.anchors[..cap]
            .iter()
            .find(|a| (a.is_active != 0) & (a.id == anchor_id))
    }

    /// Finds a mutable anchor by unique anchor ID in this lane.
    #[inline]
    pub fn get_anchor_mut(&mut self, anchor_id: u32) -> Option<&mut AttractorAnchor> {
        if anchor_id == 0 {
            return None;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        self.anchors[..cap]
            .iter_mut()
            .find(|a| (a.is_active != 0) & (a.id == anchor_id))
    }

    /// Deactivates and removes an anchor by anchor ID.
    #[inline]
    pub fn remove_anchor(&mut self, anchor_id: u32) -> bool {
        if anchor_id == 0 {
            return false;
        }
        let step = self.step_counter;
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        if let Some(slot_idx) = self.anchors[..cap]
            .iter()
            .position(|a| (a.is_active != 0) & (a.id == anchor_id))
        {
            let score = self.anchors[slot_idx].compute_energy_score(step);
            self.dissipated_energy_total += score;
            self.anchors[slot_idx].deactivate_in_place();
            self.active_count = self.active_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Register-resident core evaluator returning `(composite_words_64, pull_force)` directly in
    /// 8 `u64` SWAR registers without intermediate memory stores or reloads.
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_composite_attractor_force_words(
        &self,
        init_w: &[u64; 8],
        init_nonzero: bool,
    ) -> ([u64; 8], f32) {
        if self.active_count == 0 && self.repulsion_gain <= 0.0 {
            return (*init_w, 0.0);
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let init_is_uniform = if init_nonzero {
            is_word_array_uniform_64(init_w)
        } else {
            true
        };
        let init_w0 = init_w[0];
        let init_b0 = (init_w0 & 0xFF) as u8;
        let init_is_byte_uniform = init_is_uniform & (init_w0 == init_w0.rotate_right(8));
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let mut run_w = *init_w;
        let mut run_is_uniform = init_is_uniform;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut run_w0 = init_w0;
        let mut running_pull = 0.0_f32;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut vi_vecs: Option<(
            core::arch::x86_64::__m256i,
            core::arch::x86_64::__m256i,
        )> = if init_is_uniform {
            None
        } else {
            // SAFETY: Guarded by `target_feature = "avx2"`.
            Some(unsafe {
                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(init_w, false)
            })
        };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut vr_vecs: Option<(
            core::arch::x86_64::__m256i,
            core::arch::x86_64::__m256i,
        )> = vi_vecs;

        let mut remaining = self.active_count;
        let mut idx = 0usize;
        while idx < cap && remaining > 0 {
            let anchor = &self.anchors[idx];
            if anchor.is_active != 0 {
                remaining -= 1;
                let energy = anchor.compute_energy_score(self.step_counter);
                let base_pull = branchless_relu_f32(anchor.weight)
                    * branchless_relu_f32(anchor.pull_force)
                    * energy;
                if base_pull > DEFAULT_ENTROPY_EPS {
                    let anc_w = anchor.phase_words_ref();
                    let check_anc_uni = (init_nonzero & init_is_uniform)
                        | (running_pull <= DEFAULT_ENTROPY_EPS)
                        | run_is_uniform;
                    let anc_is_uniform = if check_anc_uni {
                        is_word_array_uniform_64(anc_w)
                    } else {
                        false
                    };
                    let anc_nonzero = if anc_is_uniform {
                        anc_w[0] != 0
                    } else {
                        swar_words_nonzero_64(anc_w)
                    };
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    if anc_is_uniform & run_is_uniform & (init_is_uniform | !init_nonzero) {
                        let aw0 = anc_w[0];
                        let eff_coh = if init_nonzero & (aw0 != 0) {
                            if aw0 == init_w0 {
                                1.0
                            } else if init_is_byte_uniform && (aw0 == aw0.rotate_right(8)) {
                                Q7_FULL_COS_REF[init_b0.wrapping_sub(aw0 as u8) as usize]
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    init_w, true, anc_w, true,
                                )
                            }
                        } else {
                            0.0_f32
                        };
                        let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
                        let pull = branchless_min_f32(
                            branchless_flush_subnormal_f32(base_pull * proximity_gain),
                            MAX_ATTRACTOR_PULL_FORCE,
                        );
                        if pull > DEFAULT_ENTROPY_EPS {
                            let total_w = running_pull + pull;
                            if running_pull <= DEFAULT_ENTROPY_EPS {
                                run_w0 = aw0;
                            } else if run_w0 != aw0 {
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    run_w0 = aw0;
                                } else if s > 0 {
                                    let bias_restore =
                                        (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                    run_w0 =
                                        crate::traits::branchless::swar_slerp_word_unchecked_x8(
                                            run_w0,
                                            aw0,
                                            s,
                                            bias_restore,
                                        );
                                }
                            }
                            running_pull = total_w;
                        }
                        idx += 1;
                        continue;
                    }
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    let (va0, va1) = unsafe {
                        crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                            anc_w,
                            anc_is_uniform,
                        )
                    };
                    let eff_coh = if init_nonzero & anc_nonzero {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            unsafe {
                                use core::arch::x86_64::_mm256_sub_epi8;
                                let (vi0, vi1) = match vi_vecs {
                                    Some(v) => v,
                                    None => {
                                        let v = crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                            init_w,
                                            init_is_uniform,
                                        );
                                        vi_vecs = Some(v);
                                        v
                                    }
                                };
                                crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vi0, va0),
                                    _mm256_sub_epi8(vi1, va1),
                                )
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            let aw0 = anc_w[0];
                            if init_is_uniform & anc_is_uniform {
                                if aw0 == init_w0 {
                                    1.0
                                } else if init_is_byte_uniform && (aw0 == aw0.rotate_right(8)) {
                                    Q7_FULL_COS_REF[init_b0.wrapping_sub(aw0 as u8) as usize]
                                } else {
                                    coherence_from_word_arrays_with_both_uniform_flags_64(
                                        init_w, true, anc_w, true,
                                    )
                                }
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    init_w,
                                    init_is_uniform,
                                    anc_w,
                                    anc_is_uniform,
                                )
                            }
                        }
                    } else {
                        0.0_f32
                    };
                    let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
                    let pull = branchless_min_f32(
                        branchless_flush_subnormal_f32(base_pull * proximity_gain),
                        MAX_ATTRACTOR_PULL_FORCE,
                    );
                    if pull > DEFAULT_ENTROPY_EPS {
                        let total_w = running_pull + pull;
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            if running_pull <= DEFAULT_ENTROPY_EPS {
                                if anc_is_uniform {
                                    run_w0 = anc_w[0];
                                    run_is_uniform = true;
                                } else {
                                    vr_vecs = Some((va0, va1));
                                    run_is_uniform = false;
                                }
                            } else {
                                let (mut vr0, mut vr1) = match vr_vecs {
                                    Some(v) if !run_is_uniform => v,
                                    _ => unsafe {
                                        let rz =
                                            core::arch::x86_64::_mm256_set1_epi64x(run_w0 as i64);
                                        (rz, rz)
                                    },
                                };
                                run_is_uniform = false;
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    vr0 = va0;
                                    vr1 = va1;
                                } else if s > 0 {
                                    unsafe {
                                        use core::arch::x86_64::{
                                            _mm256_or_si256, _mm256_sub_epi8, _mm256_testz_si256,
                                        };
                                        let vd0 = _mm256_sub_epi8(va0, vr0);
                                        let vd1 = _mm256_sub_epi8(va1, vr1);
                                        let vd_or = _mm256_or_si256(vd0, vd1);
                                        if _mm256_testz_si256(vd_or, vd_or) == 0 {
                                            (vr0, vr1) =
                                                crate::traits::branchless::avx2_slerp_from_diff_vec64(
                                                    vr0,
                                                    vr1,
                                                    vd0,
                                                    vd1,
                                                    s,
                                                );
                                        }
                                    }
                                }
                                vr_vecs = Some((vr0, vr1));
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            if running_pull <= DEFAULT_ENTROPY_EPS {
                                run_w = *anc_w;
                                run_is_uniform = anc_is_uniform;
                            } else if !crate::traits::branchless::swar_words_equal_64(&run_w, anc_w)
                            {
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                let bias_restore =
                                    (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                                let (next_w, next_uni) =
                                    crate::traits::branchless::swar_slerp_words_with_uniform_flag_64(
                                        &run_w,
                                        run_is_uniform,
                                        anc_w,
                                        anc_is_uniform,
                                        s,
                                        bias_restore,
                                    );
                                run_w = next_w;
                                run_is_uniform = next_uni;
                            }
                        }
                        running_pull = total_w;
                    }
                }
            }
            idx += 1;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut run_w = if running_pull <= DEFAULT_ENTROPY_EPS {
            *init_w
        } else if run_is_uniform {
            [run_w0; 8]
        } else if let Some((vr0, vr1)) = vr_vecs {
            unsafe { crate::traits::branchless::avx2_vec64_to_words64(vr0, vr1) }
        } else {
            [run_w0; 8]
        };

        let rep_scale_u7 = (branchless_clamp_f32(
            branchless_relu_f32(self.repulsion_gain),
            0.0,
            1.0,
        ) * 128.0
            + 0.5) as u8;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let run_is_uniform = if rep_scale_u7 > 0 {
            run_is_uniform || is_word_array_uniform_64(&run_w)
        } else {
            run_is_uniform
        };
        if rep_scale_u7 > 0 {
            let rep_w = *self.repulsion_words_ref();
            if swar_words_nonzero_64(&rep_w) {
                let rep_s = rep_scale_u7 as u64;
                let rep_bias = (128u64.wrapping_sub(rep_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                if is_word_array_uniform_64(&rep_w) {
                    let sc0 = swar_q7_scale_le128_with_bias_x8(rep_w[0], rep_s, rep_bias);
                    if sc0 != 0 {
                        if run_is_uniform {
                            run_w = [swar_q7_sub_sat_x8(run_w[0], sc0); 8];
                        } else {
                            run_w[0] = swar_q7_sub_sat_x8(run_w[0], sc0);
                            run_w[1] = swar_q7_sub_sat_x8(run_w[1], sc0);
                            run_w[2] = swar_q7_sub_sat_x8(run_w[2], sc0);
                            run_w[3] = swar_q7_sub_sat_x8(run_w[3], sc0);
                            run_w[4] = swar_q7_sub_sat_x8(run_w[4], sc0);
                            run_w[5] = swar_q7_sub_sat_x8(run_w[5], sc0);
                            run_w[6] = swar_q7_sub_sat_x8(run_w[6], sc0);
                            run_w[7] = swar_q7_sub_sat_x8(run_w[7], sc0);
                        }
                    }
                } else {
                    run_w[0] = swar_q7_sub_sat_x8(
                        run_w[0],
                        swar_q7_scale_le128_with_bias_x8(rep_w[0], rep_s, rep_bias),
                    );
                    run_w[1] = swar_q7_sub_sat_x8(
                        run_w[1],
                        swar_q7_scale_le128_with_bias_x8(rep_w[1], rep_s, rep_bias),
                    );
                    run_w[2] = swar_q7_sub_sat_x8(
                        run_w[2],
                        swar_q7_scale_le128_with_bias_x8(rep_w[2], rep_s, rep_bias),
                    );
                    run_w[3] = swar_q7_sub_sat_x8(
                        run_w[3],
                        swar_q7_scale_le128_with_bias_x8(rep_w[3], rep_s, rep_bias),
                    );
                    run_w[4] = swar_q7_sub_sat_x8(
                        run_w[4],
                        swar_q7_scale_le128_with_bias_x8(rep_w[4], rep_s, rep_bias),
                    );
                    run_w[5] = swar_q7_sub_sat_x8(
                        run_w[5],
                        swar_q7_scale_le128_with_bias_x8(rep_w[5], rep_s, rep_bias),
                    );
                    run_w[6] = swar_q7_sub_sat_x8(
                        run_w[6],
                        swar_q7_scale_le128_with_bias_x8(rep_w[6], rep_s, rep_bias),
                    );
                    run_w[7] = swar_q7_sub_sat_x8(
                        run_w[7],
                        swar_q7_scale_le128_with_bias_x8(rep_w[7], rep_s, rep_bias),
                    );
                }
            }
        }

        let pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (run_w, pull)
    }

    /// Evaluates composite gravitational pull force and 64-head phase shift for `trail` using
    /// **Register-Resident 8-Word SWAR $S^1$ Geodesic Folding** (`0 fdiv`, `0` intermediate
    /// memory reloads of `trail.accumulated_phase_delta`) and branchless SWAR repulsion subtraction.
    #[inline]
    pub fn evaluate_composite_attractor_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);
        let (run_w, pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        store_q7_words_64(composite_phase_out, &run_w);
        pull
    }

    /// Evaluates the circular statistics summary `(pull_force, r_bar, mean_phase_q7, von_mises_kappa)`
    /// of this lane's composite phase field around `trail`. As more coherent sample points are
    /// added to the lane, `r_bar` approaches `1.0` and `von_mises_kappa` increases.
    #[inline]
    #[must_use]
    pub fn evaluate_field_concentration(
        &self,
        trail: &gcso_pointer_trail_t,
    ) -> (f32, f32, gcso_q7_t, f32) {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);
        let (run_w, pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        let (r_bar, mean_q7, kappa) = circular_resultant_from_words_64(&run_w);
        (pull, r_bar, mean_q7, kappa)
    }

    /// Exports this lane's active and crystallized sample points into a compact 8,448-byte
    /// [`AttractorGrowthSnapshot`] (`Pod`) stamped with an IEEE 802.3 CRC32 checksum.
    #[inline]
    pub fn export_growth_snapshot(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        snapshot_out: &mut AttractorGrowthSnapshot,
    ) -> u32 {
        let init_w = trail_phase_words_ref(reference_trail);
        let init_nonzero = (init_w[0] != 0) || swar_words_nonzero_64(init_w);
        let (run_w, _pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        let prev_clear_end = snapshot_out.init_header_and_persona_from_words(
            self.step_counter,
            DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            1u32 << (self.domain_id & 7),
            &[1.0; MAX_ATTRACTOR_DOMAINS],
            self.repulsion_gain,
            &self.repulsion_phase_deltas,
            self.total_potential_energy(),
            &run_w,
        );

        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut remaining = self.active_count;
        let mut out_idx = 0usize;
        let mut cryst_cnt = 0u32;
        for anchor in &self.anchors[..cap] {
            if remaining == 0 || out_idx >= MAX_ANCHORS_PER_FIELD {
                break;
            }
            if anchor.is_active != 0 {
                remaining -= 1;
                snapshot_out.anchors[out_idx] = CrystallizedAnchorRecord::from_anchor(anchor);
                cryst_cnt = cryst_cnt.wrapping_add(u32::from(anchor.is_crystallized != 0));
                out_idx += 1;
            }
        }
        snapshot_out.finalize_anchors_and_crc32(out_idx, cryst_cnt, prev_clear_end)
    }

    /// Restores this lane's state from a verified [`AttractorGrowthSnapshot`].
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
        self.clear_anchors()?;
        self.step_counter = snapshot.global_step_counter;
        self.repulsion_gain = snapshot.repulsion_gain;
        self.repulsion_phase_deltas = snapshot.repulsion_phase_deltas;

        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let count = (snapshot.active_anchor_count as usize).min(cap);
        let mut max_id = 0u32;
        let mut active_loaded = 0u32;
        for rec in snapshot.anchors.iter().take(count) {
            if rec.is_active != 0 {
                rec.write_to_anchor_in_place(&mut self.anchors[active_loaded as usize], None);
                if rec.id > max_id {
                    max_id = rec.id;
                }
                active_loaded = active_loaded.wrapping_add(1);
            }
        }
        self.active_count = active_loaded;
        self.next_anchor_id = max_id.wrapping_add(1).max(1);
        Ok(())
    }
}

impl AttractorField for AttractorFieldEngine {
    #[inline(always)]
    fn anchor_count(&self) -> u32 {
        self.active_count
    }

    #[inline(always)]
    fn apply_evaporative_decay(&mut self) {
        Self::apply_evaporative_decay(self);
    }

    #[inline(always)]
    fn total_potential_energy(&self) -> f32 {
        Self::total_potential_energy(self)
    }

    #[inline(always)]
    fn pressure_ratio(&self) -> f32 {
        Self::pressure_ratio(self)
    }

    #[inline]
    fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
        let slot_idx = self.displace_weakest_anchor_slot()?;
        let anchor = &self.anchors[slot_idx];
        if anchor.is_active != 0 {
            let anchor_id = anchor.id;
            let existing_score = anchor.compute_energy_score(self.step_counter);
            self.dissipated_energy_total += existing_score;
            self.anchors[slot_idx].deactivate_in_place();
            self.active_count = self.active_count.saturating_sub(1);
            Some(anchor_id)
        } else {
            None
        }
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32> {
        if prompt.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let phases = hash_prompt_to_q7_phases(prompt, self.domain_id);
        self.insert_anchor_internal(
            GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
            weight,
            None,
            Some(&phases),
            false,
        )
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        if embedding.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.insert_anchor_internal(anchor_type, weight, Some(embedding), None, false)
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
        self.insert_anchor_internal(anchor_type, 1.0, Some(vec), None, false)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()> {
        if repulsion_deltas.is_empty() || !branchless_is_finite_f32(gain) || gain <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.repulsion_gain = gain;
        let copy_len = repulsion_deltas.len().min(MAX_ATTENTION_HEADS);
        self.repulsion_phase_deltas[..copy_len]
            .copy_from_slice(&repulsion_deltas.as_slice()[..copy_len]);
        Ok(())
    }

    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        Ok(self.evaluate_composite_attractor_force(trail, composite_phase_out))
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return Ok(0);
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut crystallized_count = 0u32;
        for anchor in &mut self.anchors[..cap] {
            if anchor.is_active != 0 {
                if (anchor.is_crystallized == 0) & (anchor.stigmergic_density > 2.0) {
                    anchor.is_crystallized = 1;
                    anchor.anchor_type = GCSO_ANCHOR_TYPE_CRYSTALLIZED;
                    anchor.pull_force =
                        branchless_min_f32(anchor.pull_force * 1.5, MAX_ATTRACTOR_PULL_FORCE);
                    crystallized_count = crystallized_count.wrapping_add(1);
                }
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        Ok(crystallized_count)
    }

    #[inline]
    fn clear_anchors(&mut self) -> GcsoResult<()> {
        let mut remaining = self.active_count;
        if remaining > 0 {
            let cap = self.anchors.len();
            let mut idx = 0usize;
            while idx < cap {
                if self.anchors[idx].is_active != 0 {
                    self.anchors[idx].deactivate_in_place();
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                idx += 1;
            }
        }
        self.active_count = 0;
        self.repulsion_gain = 0.0;
        self.repulsion_phase_deltas = [0; MAX_ATTENTION_HEADS];
        Ok(())
    }
}

impl PersonaPatcher for AttractorFieldEngine {
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
        let _ = self.insert_anchor_internal(
            GCSO_ANCHOR_TYPE_EMBEDDING,
            1.5,
            None,
            Some(&phases),
            false,
        )?;
        Ok(())
    }
}
