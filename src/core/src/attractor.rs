// name: src/core/src/attractor.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Macro Attractor Field Engine and Multi-Domain Space for Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This module decouples the Macro Attractor Field logic from the Micro Swarm module (`swarm.rs`),
//! establishing a specialized Attractor Field architecture with Accumulator Pressure Dynamics.
//!
//! # Key Architectural Concepts
//! 1. **Accumulator Pressure Dynamics**: The field continuously acts as a fixed-capacity potential accumulator
//!    striving to maintain `len() == capacity()`. Low-density or obsolete anchors are dynamically evaporated/displaced
//!    when new memory sources or system anchors enter under maximum external pressure.
//! 2. **Multi-Domain Attractor Space**: Supports isolated parallel attractor domains (Persona, General Knowledge Source,
//!    System Constraint, Anti-Hallucination Repulsion, User Interaction Context) to achieve multi-persona switching
//!    and precise Knowledge integration via topological field superposition.
//! 3. **Zero Dynamic Allocation**: Operations execute strictly within fixed-capacity structures and memory-aligned
//!    wrappers (`AlignedSlice32`, `AlignedSlice16`) guaranteeing Hot Path $\mathcal{O}(1)$ or $\mathcal{O}(K)$ bounds.

#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::too_many_lines)]

use crate::abi::{
    gcso_anchor_type_t, gcso_pointer_trail_t, gcso_q7_t, GCSO_ANCHOR_TYPE_CRYSTALLIZED,
    GCSO_ANCHOR_TYPE_EMBEDDING, GCSO_ANCHOR_TYPE_SYSTEM_PROMPT, GCSO_ANCHOR_TYPE_TOPOLOGICAL,
    GCSO_ERROR_ACTION_HUB_FULL, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::{AlignedSlice16, AlignedSlice32, AttractorField, GcsoResult};

/// Maximum anchors supported per individual attractor field domain (64 slots).
pub const MAX_ANCHORS_PER_FIELD: usize = 64;

/// Maximum dimension supported for embedding vectors in inline zero-alloc storage.
pub const MAX_EMBEDDING_DIM: usize = 128;

/// Maximum number of distinct parallel attractor domains supported in Multi-Attractor Space.
pub const MAX_ATTRACTOR_DOMAINS: usize = 8;

/// Default attractor domain identifier for System Baseline Constraints.
pub const DOMAIN_SYSTEM_BASELINE: u32 = 0;
/// Default attractor domain identifier for Persona Patches.
pub const DOMAIN_PERSONA: u32 = 1;
/// Default attractor domain identifier for Generalized Knowledge Sources.
pub const DOMAIN_KNOWLEDGE: u32 = 2;
/// Default attractor domain identifier for Anti-Hallucination Phase Repulsion.
pub const DOMAIN_PHASE_REPULSION: u32 = 3;
/// Default attractor domain identifier for User Interaction Context.
pub const DOMAIN_USER_INTERACTION: u32 = 4;

/// Single topological anchor descriptor in phase space.
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorAnchor {
    /// Unique anchor identifier (1-based index, 0 indicates unallocated slot).
    pub id: u32,
    /// Classification type of topological anchor.
    pub anchor_type: gcso_anchor_type_t,
    /// Scalar weight / baseline importance of anchor.
    pub weight: f32,
    /// Dynamic gravitational pull force exerted by anchor ($F_{\mathrm{pull}}$).
    pub pull_force: f32,
    /// Accumulated stigmergic density ($\rho_{\mathrm{stigmergy}}$).
    pub stigmergic_density: f32,
    /// Token step index when this anchor was last activated or traversed.
    pub last_active_step: u64,
    /// Half-life in token steps for temporal energy decay under accumulator pressure.
    pub half_life_steps: u32,
    /// Target phase offsets across 64 attention heads (Q7 fixed-point).
    pub phase_deltas: [gcso_q7_t; 64],
    /// Continuous feature embedding vector buffer (up to 128 dimensions).
    pub vector_data: [f32; MAX_EMBEDDING_DIM],
    /// Active dimension of stored vector data.
    pub vector_dim: u16,
    /// Flag indicating whether anchor is active.
    pub is_active: u8,
    /// Flag indicating whether anchor was bottom-up crystallized from high-density trails.
    pub is_crystallized: u8,
    /// Domain identifier associated with this anchor.
    pub domain_id: u32,
}

impl Default for AttractorAnchor {
    #[inline]
    fn default() -> Self {
        Self {
            id: 0,
            anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
            weight: 1.0,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            last_active_step: 0,
            half_life_steps: 1000,
            phase_deltas: [0; 64],
            vector_data: [0.0; MAX_EMBEDDING_DIM],
            vector_dim: 0,
            is_active: 0,
            is_crystallized: 0,
            domain_id: DOMAIN_SYSTEM_BASELINE,
        }
    }
}

impl AttractorAnchor {
    /// Computes composite energy potential score with temporal decay under Accumulator Pressure.
    #[inline(always)]
    #[must_use]
    pub fn compute_energy_score(&self, current_step: u64) -> f32 {
        if self.is_active == 0 {
            0.0
        } else {
            let dt = current_step.saturating_sub(self.last_active_step) as f32;
            let half_life = if self.half_life_steps == 0 {
                1000.0
            } else {
                self.half_life_steps as f32
            };
            // Linear-rational decay approximation for no_std efficiency: 1 / (1 + dt / half_life)
            let temporal_decay = 1.0 / (1.0 + (dt / half_life));
            let base_energy = self.weight.mul_add(self.pull_force, self.stigmergic_density);
            base_energy * temporal_decay
        }
    }
}

/// Dynamic Attractor Field Engine with Accumulator Pressure Dynamics for a single domain.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorFieldEngine {
    /// Fixed-capacity storage array for topological anchors.
    pub anchors: [AttractorAnchor; MAX_ANCHORS_PER_FIELD],
    /// Number of active anchors registered in field.
    pub active_count: u32,
    /// Maximum capacity of this field domain.
    pub capacity: u32,
    /// Target domain identifier.
    pub domain_id: u32,
    /// Global phase repulsion gain factor.
    pub repulsion_gain: f32,
    /// Global anti-phase repulsion profile vector ($-\boldsymbol{\Delta\theta}$).
    pub repulsion_phase_deltas: [gcso_q7_t; 64],
    /// Accumulator pressure factor controlling minimum threshold for evaporative displacement.
    pub accumulator_pressure: f32,
    /// Evaporation rate per step applied to stigmergic density under external pressure.
    pub evaporation_rate: f32,
    /// Internal token step counter for LRU and decay tracking.
    pub step_counter: u64,
    /// Auto-incrementing anchor ID generator.
    next_anchor_id: u32,
}

impl Default for AttractorFieldEngine {
    #[inline]
    fn default() -> Self {
        Self::new(DOMAIN_SYSTEM_BASELINE, MAX_ANCHORS_PER_FIELD as u32)
    }
}

impl AttractorFieldEngine {
    /// Creates a new `AttractorFieldEngine` instance for a specific domain.
    #[inline]
    #[must_use]
    pub fn new(domain_id: u32, capacity: u32) -> Self {
        let cap = (capacity as usize).min(MAX_ANCHORS_PER_FIELD) as u32;
        Self {
            anchors: [AttractorAnchor::default(); MAX_ANCHORS_PER_FIELD],
            active_count: 0,
            capacity: cap,
            domain_id,
            repulsion_gain: 0.0,
            repulsion_phase_deltas: [0; 64],
            accumulator_pressure: 0.5,
            evaporation_rate: 0.01,
            step_counter: 0,
            next_anchor_id: 1,
        }
    }

    /// Returns the active anchor count.
    #[inline(always)]
    #[must_use]
    pub const fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Returns the domain identifier.
    #[inline(always)]
    #[must_use]
    pub const fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Returns `true` if field is saturated (`active_count == capacity`).
    #[inline(always)]
    #[must_use]
    pub const fn is_saturated(&self) -> bool {
        self.active_count >= self.capacity
    }

    /// Evaluates evaporative pressure decay across all active anchors.
    ///
    /// Automatically deactivates non-crystallized anchors whose energy drops below the pressure threshold.
    #[inline]
    pub fn apply_evaporative_decay(&mut self) {
        if self.active_count == 0 {
            return;
        }

        let mut deactivated_count = 0u32;
        for anchor in self.anchors.iter_mut().take(self.capacity as usize) {
            if anchor.is_active == 0 || anchor.is_crystallized != 0 {
                continue;
            }

            // Apply density evaporation
            anchor.stigmergic_density *= 1.0 - self.evaporation_rate;

            let current_energy = anchor.compute_energy_score(self.step_counter);
            if current_energy < (0.05 * self.accumulator_pressure) {
                *anchor = AttractorAnchor::default();
                deactivated_count = deactivated_count.wrapping_add(1);
            }
        }

        self.active_count = self.active_count.saturating_sub(deactivated_count);
    }

    /// Evaporates/displaces the weakest non-crystallized anchor when accumulator pressure forces replacement.
    ///
    /// Finds slot with lowest composite energy score. If all are crystallized, falls back to LRU displacement.
    #[inline]
    fn displace_weakest_anchor_slot(&mut self) -> Option<usize> {
        let mut lowest_score = f32::MAX;
        let mut target_idx = None;
        let mut oldest_step = u64::MAX;
        let mut lru_idx = None;

        for (idx, anchor) in self.anchors.iter().enumerate().take(self.capacity as usize) {
            if anchor.is_active == 0 {
                return Some(idx);
            }

            // Crystallized anchors are protected from energy displacement unless LRU forced
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

        // Return lowest energy non-crystallized slot, or oldest LRU slot if all are crystallized
        target_idx.or(lru_idx)
    }

    /// Allocates an anchor slot enforcing Accumulator Pressure Dynamics (Evaporative Displacement).
    #[inline]
    fn allocate_anchor_slot(&mut self, incoming_energy: f32) -> GcsoResult<usize> {
        if !self.is_saturated() {
            for (idx, anchor) in self.anchors.iter().enumerate().take(self.capacity as usize) {
                if anchor.is_active == 0 {
                    self.active_count = self.active_count.wrapping_add(1);
                    return Ok(idx);
                }
            }
        }

        // Field is saturated: Apply Accumulator Pressure to displace weakest anchor
        let slot_idx = self
            .displace_weakest_anchor_slot()
            .ok_or(GCSO_ERROR_ACTION_HUB_FULL)?;

        let target_anchor = &self.anchors[slot_idx];
        if target_anchor.is_active != 0 {
            let existing_score = target_anchor.compute_energy_score(self.step_counter);
            // Displace only if incoming anchor energy meets accumulator pressure threshold
            if incoming_energy < existing_score * self.accumulator_pressure {
                return Err(GCSO_ERROR_ACTION_HUB_FULL);
            }
        }

        Ok(slot_idx)
    }

    /// Internal helper to register an anchor into an allocated slot.
    #[inline]
    fn insert_anchor_internal(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
        embedding: Option<&AlignedSlice32<'_, f32>>,
        is_crystallized: bool,
    ) -> GcsoResult<u32> {
        if weight.is_nan() || weight.is_infinite() || weight <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let initial_energy = weight * 1.0; // Baseline initial energy
        let slot_idx = self.allocate_anchor_slot(initial_energy)?;

        let anchor_id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.wrapping_add(1);
        if self.next_anchor_id == 0 {
            self.next_anchor_id = 1;
        }

        let slot = &mut self.anchors[slot_idx];
        *slot = AttractorAnchor::default();
        slot.id = anchor_id;
        slot.anchor_type = anchor_type;
        slot.weight = weight;
        slot.pull_force = 1.0;
        slot.stigmergic_density = 1.0;
        slot.last_active_step = self.step_counter;
        slot.half_life_steps = 1000;
        slot.is_active = 1;
        slot.is_crystallized = u8::from(is_crystallized);
        slot.domain_id = self.domain_id;

        if let Some(emb) = embedding {
            let copy_len = emb.len().min(MAX_EMBEDDING_DIM);
            if copy_len > 0 {
                slot.vector_data[..copy_len].copy_from_slice(&emb.as_slice()[..copy_len]);
                slot.vector_dim = copy_len as u16;

                // Generate initial phase profile from embedding dimensions safely
                for (h, delta) in slot.phase_deltas.iter_mut().enumerate() {
                    let sample_val = slot.vector_data[h % copy_len];
                    let q7_val = (sample_val * 128.0) as i32;
                    *delta = q7_val.clamp(-128, 127) as i8;
                }
            }
        }

        Ok(anchor_id)
    }

    /// Calculates composite gravitational pull force and combined phase shift for a given trail.
    #[inline]
    pub fn evaluate_composite_attractor_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; 64],
    ) -> f32 {
        let mut total_pull = 0.0f32;
        let mut phase_acc = [0i32; 64];

        for anchor in self.anchors.iter().take(self.capacity as usize) {
            if anchor.is_active == 0 {
                continue;
            }

            let energy_score = anchor.compute_energy_score(self.step_counter);
            let pull = anchor.weight * anchor.pull_force * energy_score;
            total_pull += pull;

            let pull_weight = (pull * 128.0) as i32;
            for (h, &phase) in anchor.phase_deltas.iter().enumerate() {
                phase_acc[h] += (phase as i32) * pull_weight;
            }
        }

        if total_pull > 0.0 {
            let norm = (total_pull * 128.0) as i32;
            if norm > 0 {
                for (h, acc) in phase_acc.iter().enumerate() {
                    let avg_q7 = acc / norm;
                    composite_phase_out[h] = avg_q7.clamp(-128, 127) as i8;
                }
            }
        } else {
            composite_phase_out.copy_from_slice(&trail.accumulated_phase_delta);
        }

        // Apply phase repulsion pulse if active in this field
        if self.repulsion_gain > 0.0 {
            let rep_scale = (self.repulsion_gain * 128.0) as i32;
            for (h, &rep_phase) in self.repulsion_phase_deltas.iter().enumerate() {
                let current = composite_phase_out[h] as i32;
                let rep_adj = (rep_phase as i32 * rep_scale) >> 7;
                composite_phase_out[h] = (current - rep_adj).clamp(-128, 127) as i8;
            }
        }

        total_pull
    }
}

impl AttractorField for AttractorFieldEngine {
    #[inline(always)]
    fn anchor_count(&self) -> u32 {
        self.active_count
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32> {
        if prompt.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.insert_anchor_internal(GCSO_ANCHOR_TYPE_SYSTEM_PROMPT, weight, None, false)
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
        self.insert_anchor_internal(anchor_type, weight, Some(embedding), false)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32> {
        if dim == 0 || vec.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.insert_anchor_internal(anchor_type, 1.0, Some(vec), false)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()> {
        if gain.is_nan() || gain.is_infinite() || gain <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.repulsion_gain = gain;
        let copy_len = repulsion_deltas.len().min(64);
        self.repulsion_phase_deltas[..copy_len]
            .copy_from_slice(&repulsion_deltas.as_slice()[..copy_len]);
        Ok(())
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
        // Bottom-up macro crystallization: convert high-density anchors into crystallized status
        let mut crystallized_count = 0u32;
        for anchor in self.anchors.iter_mut().take(self.capacity as usize) {
            if anchor.is_active != 0
                && anchor.is_crystallized == 0
                && anchor.stigmergic_density > 2.0
            {
                anchor.is_crystallized = 1;
                anchor.anchor_type = GCSO_ANCHOR_TYPE_CRYSTALLIZED;
                anchor.pull_force *= 1.5; // Boost pull force for crystallized invariants
                crystallized_count = crystallized_count.wrapping_add(1);
            }
        }
        Ok(crystallized_count)
    }

    #[inline]
    fn clear_anchors(&mut self) -> GcsoResult<()> {
        self.anchors.fill(AttractorAnchor::default());
        self.active_count = 0;
        self.repulsion_gain = 0.0;
        self.repulsion_phase_deltas.fill(0);
        Ok(())
    }
}

/// Multi-Domain Attractor Space managing isolated parallel attractor domains
/// (Persona, Knowledge Sources, System Baseline, Phase Repulsion, User Context).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiAttractorSpace {
    /// Domain 0: System Baseline Attractor Field.
    pub system_field: AttractorFieldEngine,
    /// Domain 1: Active Persona Patch Attractor Field.
    pub persona_field: AttractorFieldEngine,
    /// Domain 2: General Knowledge Source Attractor Field.
    pub knowledge_field: AttractorFieldEngine,
    /// Domain 3: Anti-Hallucination Phase Repulsion Field.
    pub repulsion_field: AttractorFieldEngine,
    /// Domain 4: User Interaction Context Attractor Field.
    pub user_field: AttractorFieldEngine,
    /// Relative weighting multipliers for each domain (up to 8 domains).
    pub domain_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Active domain selection bitmask.
    pub active_domains_mask: u32,
    /// Global token step counter across space.
    pub global_step_counter: u64,
}

impl Default for MultiAttractorSpace {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAttractorSpace {
    /// Creates a new `MultiAttractorSpace` with initialized domain fields.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        let mut domain_weights = [1.0f32; MAX_ATTRACTOR_DOMAINS];
        domain_weights[DOMAIN_SYSTEM_BASELINE as usize] = 1.0;
        domain_weights[DOMAIN_PERSONA as usize] = 1.2;
        domain_weights[DOMAIN_KNOWLEDGE as usize] = 1.5;
        domain_weights[DOMAIN_PHASE_REPULSION as usize] = 1.0;
        domain_weights[DOMAIN_USER_INTERACTION as usize] = 1.1;

        Self {
            system_field: AttractorFieldEngine::new(
                DOMAIN_SYSTEM_BASELINE,
                MAX_ANCHORS_PER_FIELD as u32,
            ),
            persona_field: AttractorFieldEngine::new(DOMAIN_PERSONA, 32),
            knowledge_field: AttractorFieldEngine::new(DOMAIN_KNOWLEDGE, 64),
            repulsion_field: AttractorFieldEngine::new(DOMAIN_PHASE_REPULSION, 16),
            user_field: AttractorFieldEngine::new(DOMAIN_USER_INTERACTION, 32),
            domain_weights,
            active_domains_mask: 0x1F, // Enable all 5 primary domains by default
            global_step_counter: 0,
        }
    }

    /// Advances global step counter across all domain fields and evaluates evaporative decay.
    #[inline]
    pub fn advance_step(&mut self) {
        self.global_step_counter = self.global_step_counter.wrapping_add(1);

        let fields = [
            &mut self.system_field,
            &mut self.persona_field,
            &mut self.knowledge_field,
            &mut self.repulsion_field,
            &mut self.user_field,
        ];

        for field in fields {
            field.step_counter = self.global_step_counter;
            field.apply_evaporative_decay();
        }
    }

    /// Sets relative domain weighting scalar for multi-domain field superposition.
    #[inline]
    pub fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS
            || weight.is_nan()
            || weight.is_infinite()
            || weight < 0.0
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.domain_weights[domain_id as usize] = weight;
        Ok(())
    }

    /// Retrieves a mutable reference to a specific domain field by domain ID.
    #[inline]
    pub fn domain_mut(&mut self, domain_id: u32) -> GcsoResult<&mut AttractorFieldEngine> {
        match domain_id {
            DOMAIN_SYSTEM_BASELINE => Ok(&mut self.system_field),
            DOMAIN_PERSONA => Ok(&mut self.persona_field),
            DOMAIN_KNOWLEDGE => Ok(&mut self.knowledge_field),
            DOMAIN_PHASE_REPULSION => Ok(&mut self.repulsion_field),
            DOMAIN_USER_INTERACTION => Ok(&mut self.user_field),
            _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
        }
    }

    /// Retrieves an immutable reference to a specific domain field by domain ID.
    #[inline]
    pub fn domain(&self, domain_id: u32) -> GcsoResult<&AttractorFieldEngine> {
        match domain_id {
            DOMAIN_SYSTEM_BASELINE => Ok(&self.system_field),
            DOMAIN_PERSONA => Ok(&self.persona_field),
            DOMAIN_KNOWLEDGE => Ok(&self.knowledge_field),
            DOMAIN_PHASE_REPULSION => Ok(&self.repulsion_field),
            DOMAIN_USER_INTERACTION => Ok(&self.user_field),
            _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
        }
    }

    /// Evaluates multi-domain superposition of gravitational pulls and composite phase shifts.
    ///
    /// Combines Persona, General Knowledge, System baseline, User Context, and Anti-Hallucination Repulsion
    /// potentials into a unified phase vector in zero-allocation mode.
    #[inline]
    pub fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; 64],
    ) -> f32 {
        let mut total_pull = 0.0f32;
        let mut phase_accum = [0i32; 64];

        let domains = [
            &self.system_field,
            &self.persona_field,
            &self.knowledge_field,
            &self.repulsion_field,
            &self.user_field,
        ];

        for (idx, field) in domains.iter().enumerate() {
            if (self.active_domains_mask & (1 << idx)) == 0 || field.active_count == 0 {
                continue;
            }

            let domain_weight = self.domain_weights[idx];
            if domain_weight <= 0.0 {
                continue;
            }

            let mut domain_phase = [0i8; 64];
            let raw_pull = field.evaluate_composite_attractor_force(trail, &mut domain_phase);
            let domain_pull = raw_pull * domain_weight;

            if domain_pull > 0.0 {
                total_pull += domain_pull;
                let weight_q7 = (domain_pull * 128.0) as i32;
                for (h, &p) in domain_phase.iter().enumerate() {
                    phase_accum[h] += (p as i32) * weight_q7;
                }
            }
        }

        if total_pull > 0.0 {
            let norm = (total_pull * 128.0) as i32;
            if norm > 0 {
                for (h, acc) in phase_accum.iter().enumerate() {
                    composite_phase_out[h] = (acc / norm).clamp(-128, 127) as i8;
                }
            }
        } else {
            composite_phase_out.copy_from_slice(&trail.accumulated_phase_delta);
        }

        total_pull
    }

    /// Registers a generalized knowledge source embedding into the specialized Knowledge attractor domain.
    #[inline]
    pub fn register_knowledge_source_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        source_weight: f32,
    ) -> GcsoResult<u32> {
        self.knowledge_field.add_embedding_anchor(
            embedding,
            GCSO_ANCHOR_TYPE_TOPOLOGICAL,
            source_weight,
        )
    }

    /// Registers a persona modulation patch into the isolated Persona domain.
    #[inline]
    pub fn register_persona_patch_anchor(
        &mut self,
        persona_vector: &AlignedSlice32<'_, f32>,
        persona_weight: f32,
    ) -> GcsoResult<u32> {
        self.persona_field.add_embedding_anchor(
            persona_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            persona_weight,
        )
    }

    /// Registers a user interaction context vector into the User Interaction domain.
    #[inline]
    pub fn register_user_context_anchor(
        &mut self,
        context_vector: &AlignedSlice32<'_, f32>,
        context_weight: f32,
    ) -> GcsoResult<u32> {
        self.user_field.add_embedding_anchor(
            context_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            context_weight,
        )
    }

    /// Clears all anchors across all active attractor domains.
    #[inline]
    pub fn clear_all_domains(&mut self) -> GcsoResult<()> {
        self.system_field.clear_anchors()?;
        self.persona_field.clear_anchors()?;
        self.knowledge_field.clear_anchors()?;
        self.repulsion_field.clear_anchors()?;
        self.user_field.clear_anchors()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attractor_anchor_energy_score_and_decay() {
        let mut anchor = AttractorAnchor::default();
        anchor.is_active = 1;
        anchor.weight = 2.0;
        anchor.pull_force = 1.5;
        anchor.stigmergic_density = 1.0;
        anchor.last_active_step = 0;
        anchor.half_life_steps = 100;

        // At dt = 0: 2.0 * 1.5 + 1.0 = 4.0
        assert!((anchor.compute_energy_score(0) - 4.0).abs() < 1e-5);

        // At dt = 100 (half-life): 4.0 * (1 / (1 + 100/100)) = 2.0
        assert!((anchor.compute_energy_score(100) - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_accumulator_pressure_evaporative_displacement() {
        let mut engine = AttractorFieldEngine::new(0, 3); // Capacity = 3

        #[repr(align(32))]
        struct AlignedBuf([f32; 16]);
        let buf = AlignedBuf([1.0; 16]);
        let slice = AlignedSlice32::new(&buf.0).unwrap();

        // Fill capacity (3 anchors)
        let id1 = engine.add_system_prompt_anchor("Anchor 1", 1.0).unwrap();
        let id2 = engine
            .add_embedding_anchor(&slice, GCSO_ANCHOR_TYPE_EMBEDDING, 0.5)
            .unwrap();
        let _id3 = engine.add_system_prompt_anchor("Anchor 3", 2.0).unwrap();

        assert_eq!(engine.active_count(), 3);
        assert!(engine.is_saturated());

        // Attempting to add a low-energy anchor should fail under pressure
        let err_res = engine.add_system_prompt_anchor("Weak Anchor", 0.01);
        assert_eq!(err_res, Err(GCSO_ERROR_ACTION_HUB_FULL));

        // Adding a high-energy anchor should displace the weakest anchor (id2 with weight 0.5)
        let id4 = engine
            .add_system_prompt_anchor("Strong Anchor 4", 5.0)
            .unwrap();
        assert_eq!(engine.active_count(), 3); // Capacity remains 3
        assert_ne!(id4, id1);
        assert_ne!(id4, id2);

        // Verify id2 was displaced
        let mut found_id2 = false;
        for a in engine.anchors.iter().take(3) {
            if a.id == id2 {
                found_id2 = true;
            }
        }
        assert!(!found_id2);
    }

    #[test]
    fn test_evaporative_decay_deactivation() {
        let mut engine = AttractorFieldEngine::new(0, 5);
        let _id = engine.add_system_prompt_anchor("Weak Trail", 0.1).unwrap();
        assert_eq!(engine.active_count(), 1);

        // Advance steps significantly to trigger temporal decay below threshold
        engine.step_counter = 100_000;
        engine.apply_evaporative_decay();

        assert_eq!(engine.active_count(), 0);
    }

    #[test]
    fn test_bottom_up_crystallization() {
        let mut engine = AttractorFieldEngine::new(0, 10);
        let _id = engine.add_system_prompt_anchor("Trail Anchor", 1.0).unwrap();

        // Increase stigmergic density beyond threshold
        engine.anchors[0].stigmergic_density = 3.0;

        let count = engine.aggregate_bottom_up().unwrap();
        assert_eq!(count, 1);
        assert_eq!(engine.anchors[0].is_crystallized, 1);
        assert_eq!(engine.anchors[0].anchor_type, GCSO_ANCHOR_TYPE_CRYSTALLIZED);
        assert!((engine.anchors[0].pull_force - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_multi_attractor_space_superposition() {
        let mut space = MultiAttractorSpace::new();

        #[repr(align(32))]
        struct AlignedBuf([f32; 16]);
        let buf_knowledge = AlignedBuf([0.5; 16]);
        let buf_persona = AlignedBuf([0.8; 16]);

        let slice_knowledge = AlignedSlice32::new(&buf_knowledge.0).unwrap();
        let slice_persona = AlignedSlice32::new(&buf_persona.0).unwrap();

        assert!(space.register_knowledge_source_anchor(&slice_knowledge, 2.0).is_ok());
        assert!(space
            .register_persona_patch_anchor(&slice_persona, 1.5)
            .is_ok());

        assert_eq!(space.knowledge_field.active_count(), 1);
        assert_eq!(space.persona_field.active_count(), 1);

        let trail = gcso_pointer_trail_t::default();
        let mut composite_phase = [0i8; 64];
        let total_pull = space.evaluate_multi_domain_superposition(&trail, &mut composite_phase);

        assert!(total_pull > 0.0);
    }
}