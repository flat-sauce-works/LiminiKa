// name: src/core/src/stigmergy/substrate/cascade.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pheromone evaporation, pointer-chain cascade orchestration, and trait implementations for [`StigmergicSubstrate`].

use super::{
    SUBSTRATE_INLINE_TRAILS, StigmergicActionDecision, StigmergicSubstrate, StigmergicThresholds,
    SubstrateStepReport,
};
use crate::foundation::{
    ACTION_MODE_RANK1_INJECT, ATTRACTOR_SLOT_ID_MASK, AlignedSliceMut32,
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD, PagedBitmask, SWAR_SPLAT_BYTE_64,
    TaggedPtr, branchless_clamp_f32, branchless_fast_transition_log_prob,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_relu_f32, coherence_from_word_arrays_64, fast_inv_len_f32,
    load_q7_words_64, pack_action_slot_tag, resolve_autonomous_swarm_rot_dim,
    swar_or_avx2_q7_add_sat_words_64, swar_words_nonzero_64,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    NUM_STIGMERGIC_GRANULES, PhaseQ7, RuntimeConfig, branchless_select_f32, branchless_select_u8,
    branchless_select_u32, branchless_select_u64, branchless_select_usize,
};
use crate::stigmergy::ledger::{FastPathLedger, LedgerSlot};
use crate::stigmergy::pheromone::PheromoneDensityOps;
use crate::stigmergy::trail::{
    PointerSubstrate, PointerTrail, StigmergicSubstrateOps,
    select_rewire_bridge_and_modulate_with_preloaded_words_64, trail_phase_words_mut,
    trail_phase_words_ref,
};
use crate::stigmergy::{
    LEDGER_HISTORY_LEN, LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_LAYER_SKIP_COMPENSATED,
    LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_RANK1_INJECTED, LEDGER_METRIC_REPULSION_APPLIED,
    LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, LEDGER_METRIC_SUBHEAD_ROUTED, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_INERTIAL_SKIP_ACTIVE, TRAIL_FLAG_REPULSION_ACTIVE,
};

impl StigmergicSubstrate {
    /// Applies per-step pheromone evaporation ($\rho_j \gets \rho_j \cdot \lambda_{\mathrm{retain}}$)
    /// across all active inline trails in `self.trails`, deposits pheromone reinforcement
    /// $\Delta\rho = g_{\mathrm{reinforce}} \cdot (1 + \max(0, \mathcal{C}_{\mathrm{phase}}))$ onto
    /// `trail_inout` (or halves $\rho$ on contradiction), and synchronizes `trail_inout` into its
    /// inline table slot in $\mathcal{O}(1)$.
    ///
    /// Returns `(slot_idx, updated_density)`.
    #[inline]
    pub fn evaporate_and_deposit_pheromone(
        &mut self,
        trail_inout: &mut PointerTrail,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> (u32, f32) {
        let retain = branchless_clamp_f32(
            branchless_relu_f32(self.thresholds.evaporation_retain),
            0.0,
            1.0,
        );
        let active_len = self.inline_active_len();
        let mut i = 0usize;
        while i < active_len {
            let _ = self.trails[i].evaporate_pheromone(retain);
            i += 1;
        }

        let next_rho = trail_inout.step_pheromone(
            retain,
            self.thresholds.reinforcement_gain,
            phase_coherence,
            contradiction_repulsed,
        );

        let slot_idx = self.write_cursor & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        Self::copy_trail_to_slot_fast(&mut self.trails[slot_idx as usize], trail_inout);
        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        (slot_idx, next_rho)
    }

    /// Copies `src` and pre-loaded `phase_words` into `dst` using width-matched scalar writes and
    /// 64-byte aligned 64-bit GPR stores (`*trail_phase_words_mut(dst) = *phase_words`) at byte
    /// offset `64..128`, avoiding unaligned 256-bit `vmovdqu` cache-line splits, 64-bit-to-256-bit
    /// STLF stalls, and redundant 64-byte reloads from `src`.
    #[inline(always)]
    pub(crate) fn copy_trail_and_words_to_slot_fast(
        dst: &mut PointerTrail,
        src: &PointerTrail,
        phase_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) {
        dst.copy_from_with_words_fast(src, phase_words);
    }

    #[inline(always)]
    pub(crate) fn copy_trail_to_slot_fast(dst: &mut PointerTrail, src: &PointerTrail) {
        dst.copy_from_fast(src);
    }

    /// Clears a single inline trail slot in-place using width-matched 64-bit GPR stores (`0` YMM
    /// stores, `0` 256-bit-to-64-bit STLF stalls on subsequent trail reads/writes).
    #[inline(always)]
    pub(crate) fn clear_trail_slot_fast(dst: &mut PointerTrail) {
        dst.clear_fast();
    }

    /// Applies pheromone evaporation across active inline trails (skipping `slot_idx`, which is
    /// about to be overwritten, and skipping the loop entirely when `retain >= 1.0`) and updates
    /// `trail_inout.stigmergic_density` in registers, returning `(slot_idx, updated_density)`.
    #[inline(always)]
    fn evaporate_and_prepare_slot_without_copy(
        &mut self,
        trail_inout: &mut PointerTrail,
        phase_coherence: f32,
        contradiction_repulsed: bool,
    ) -> (u32, f32) {
        let retain = branchless_clamp_f32(
            branchless_relu_f32(self.thresholds.evaporation_retain),
            0.0,
            1.0,
        );
        let slot_idx = self.write_cursor & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        let slot_usize = slot_idx as usize;
        let active_len = self.inline_active_len();
        if retain < 1.0 - 1.0e-7 {
            let mut i = 0usize;
            while i < active_len {
                if i != slot_usize {
                    let decayed = branchless_flush_subnormal_f32(
                        branchless_relu_f32(self.trails[i].stigmergic_density) * retain,
                    );
                    self.trails[i].stigmergic_density = decayed;
                }
                i += 1;
            }
        }

        let cur_rho = branchless_flush_subnormal_f32(
            branchless_relu_f32(trail_inout.stigmergic_density) * retain,
        );
        let pos_coh = branchless_clamp_f32(branchless_relu_f32(phase_coherence), 0.0, 1.0);
        let deposit = branchless_relu_f32(self.thresholds.reinforcement_gain) * (1.0 + pos_coh);
        let reinforced_rho = branchless_min_f32(cur_rho + deposit, MAX_STIGMERGIC_DENSITY);
        let damped_rho = branchless_flush_subnormal_f32(cur_rho * 0.5);
        let next_rho = branchless_select_f32(contradiction_repulsed, damped_rho, reinforced_rho);
        trail_inout.stigmergic_density = next_rho;

        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        (slot_idx, next_rho)
    }

    /// Executes the primary **Stigmergic Threshold Action-Selecting Orchestration Step** in
    /// zero-allocation constant-time mode ($C_{\mathrm{alloc}} = 0$):
    /// 1. Resolves FastPathLedger fast-path shortcut (`resolve_or_pass_tagged_ptr`).
    /// 2. Advances `trail_inout` pointer & accumulates embedded Q7 phase.
    /// 3. Evaluates `self.bloom_filter` contrastive anti-phase repulsion (`apply_phase_conjugate_repulsion_words`).
    /// 4. Scans populated inline trails `self.trails[..active_len]` via
    ///    `select_rewire_bridge_and_modulate_with_preloaded_words_64` to select the highest-scoring
    ///    non-contradictory trail $j^*$ under the Bayes-Attention transition score, rewires
    ///    `trail_inout.linked_trail_id = j^*`, bridges the inter-memory **Limina** gap via $S^1$
    ///    geodesic SLERP, and optionally modulates `attention_scores` in-place.
    /// 5. Applies global pheromone evaporation & reinforcement (`evaporate_and_deposit_pheromone`).
    /// 6. Evaluates `self.thresholds.select_action_decision` to choose the next
    ///    [`StigmergicActionDecision`] (`PassiveSidecar`, `InertialLayerSkip`, `ExpressPhaseSteer`,
    ///    `SubHeadRoute`, `FullOrchestration`, `CrystallizeMacroBasin`, or `RepulsionEscape`).
    /// 7. Executes the selected action on `query_tensor` / `key_tensor` / `logits_opt` and logs
    ///    telemetry into `self.ledger_slot`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias or have
    /// fewer than `head_dim * num_heads` elements.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn orchestrate_next_action_step(
        &mut self,
        input_ptr: TaggedPtr,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut PointerTrail,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<SubstrateStepReport> {
        self.orchestrate_next_action_step_core(
            input_ptr,
            entropy_h_tilde,
            dominant_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            true,
            None,
            DEFAULT_PHASE_STEP_MIN_RAD,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
        )
    }

    #[inline]
    // Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn orchestrate_next_action_step_core(
        &mut self,
        input_ptr: TaggedPtr,
        entropy_h_tilde: f32,
        dominant_slot: u8,
        trail_inout: &mut PointerTrail,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
        accumulate_ptr_phase: bool,
        macro_centroid_words: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
        phase_step_min_rad: f32,
        phase_clamp_max: f32,
    ) -> LmnkResult<SubstrateStepReport> {
        let (head_dim, num_heads, _expected_len) = self.validate_substrate_cascade_params(
            u32::from(self.head_dim),
            u32::from(self.num_heads),
            phase_step_min_rad,
            phase_clamp_max,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        if let Some(ref l_ref) = logits_opt {
            if let Some(ref q_ref) = query_tensor {
                q_ref.ensure_disjoint_mut(l_ref)?;
            }
            if let Some(ref k_ref) = key_tensor {
                k_ref.ensure_disjoint_mut(l_ref)?;
            }
        }

        let (bypass_hit, eff_ptr) = self.ledger_slot.resolve_or_pass_in_place(input_ptr);

        self.step_counter = self.step_counter.wrapping_add(1);
        trail_inout.record_pointer_transition(eff_ptr.raw(), 1, 0.0);
        if accumulate_ptr_phase {
            let embedded_phase = eff_ptr.unpack_phase();
            if embedded_phase != 0 {
                trail_inout.accumulate_uniform_phase(embedded_phase);
            }
        }

        let (bloom_density, bloom_repulsed) =
            self.bloom_filter.apply_phase_conjugate_repulsion_words(
                trail_phase_words_mut(trail_inout),
                self.repulsion_threshold,
                self.repulsion_scale_u7,
            );

        let pref_slot = branchless_select_u8(
            dominant_slot < (MAX_ATTRACTOR_DOMAINS as u8),
            dominant_slot,
            eff_ptr.target_attractor_slot(),
        ) & ATTRACTOR_SLOT_ID_MASK;

        let active_len = self.inline_active_len();
        let init_w = *trail_phase_words_ref(trail_inout);
        let (selected_id, best_coh, log_prob, found_coherent, mut steer_words) = if active_len > 0 {
            let scores_slice_opt = attention_scores.map(AlignedSliceMut32::as_mut_slice);
            let (sel_id, _bayes_score, _mean_coh, _coh_cnt, found, bridged_w, raw_best_coh) =
                select_rewire_bridge_and_modulate_with_preloaded_words_64(
                    trail_inout,
                    init_w,
                    &self.trails[..active_len],
                    scores_slice_opt,
                    head_dim as u32,
                    2.0,
                    0.5,
                    0.25,
                    self.thresholds.contradiction_coherence_thresh,
                    64,
                    1.0,
                    trail_inout.target_anchor_id,
                    macro_centroid_words,
                    0.25,
                );
            let lp = branchless_fast_transition_log_prob(raw_best_coh);
            (sel_id, raw_best_coh, lp, found, bridged_w)
        } else {
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `pref_slot < 8`.
            let pref_slot_w = unsafe {
                &*(self
                    .slot_phase_bank
                    .as_slice()
                    .as_ptr()
                    .add(usize::from(pref_slot) * MAX_ATTENTION_HEADS)
                    as *const [u64; NUM_STIGMERGIC_GRANULES])
            };
            let pref_enabled = ((self.slot_enabled_mask >> pref_slot) & 1) != 0;
            let slot_coh = if pref_enabled && swar_words_nonzero_64(pref_slot_w) {
                coherence_from_word_arrays_64(&init_w, pref_slot_w)
            } else {
                1.0_f32
            };
            let lp = branchless_fast_transition_log_prob(slot_coh);
            (
                trail_inout.linked_trail_id,
                slot_coh,
                lp,
                slot_coh >= -self.thresholds.contradiction_coherence_thresh,
                init_w,
            )
        };

        let contradiction_repulsed = bloom_repulsed
            | ((!found_coherent) & (best_coh < -self.thresholds.contradiction_coherence_thresh));
        if contradiction_repulsed && !bloom_repulsed {
            self.bloom_filter
                .insert_repulsive_pattern_words_weighted(&steer_words, 1);
        }

        let (stored_slot_idx, updated_rho) = self.evaporate_and_prepare_slot_without_copy(
            trail_inout,
            best_coh,
            contradiction_repulsed,
        );
        let (decision, mode_flags, target_slot, extra_flags) =
            self.thresholds.select_action_decision(
                updated_rho,
                entropy_h_tilde,
                best_coh,
                contradiction_repulsed,
                pref_slot,
            );

        let base_evolved = TaggedPtr::pack(
            eff_ptr.unpack_addr(),
            pack_action_slot_tag(target_slot, mode_flags),
            eff_ptr.unpack_phase(),
        );
        let evolved_ptr = TaggedPtr(branchless_select_u64(
            decision as u8 == StigmergicActionDecision::RepulsionEscape as u8,
            base_evolved.conjugate_phase().raw(),
            base_evolved.raw(),
        ));

        trail_inout.current_ptr = evolved_ptr.raw();
        trail_inout.cluster_id = branchless_select_u32(
            (trail_inout.target_anchor_id != 0) & (!contradiction_repulsed),
            trail_inout.target_anchor_id,
            u32::from(target_slot),
        );
        trail_inout.flags = (trail_inout.flags
            & !(TRAIL_FLAG_REPULSION_ACTIVE | TRAIL_FLAG_INERTIAL_SKIP_ACTIVE))
            | extra_flags;
        trail_inout.sync_flags_with_current_ptr();

        // Fold coupled working-lane phase profile from `slot_phase_bank[target_slot]` strictly
        // through the 4-quadrant role partition (global repulsion on contradiction, standard
        // attractive otherwise; sandboxed and sandbox-repulsion lanes never leak here):
        let masked_target_slot = target_slot & ATTRACTOR_SLOT_ID_MASK;
        let (rep_cut, attract_cut) = self.lane_role_quadrants();
        let allowed_mask =
            branchless_select_u8(contradiction_repulsed, rep_cut.outside, attract_cut.outside);
        if ((allowed_mask >> masked_target_slot) & 1) != 0 {
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `masked_target_slot < 8`.
            let slot_w = unsafe {
                &*(self
                    .slot_phase_bank
                    .as_slice()
                    .as_ptr()
                    .add(usize::from(masked_target_slot) * MAX_ATTENTION_HEADS)
                    as *const [u64; NUM_STIGMERGIC_GRANULES])
            };
            if swar_words_nonzero_64(slot_w) {
                swar_or_avx2_q7_add_sat_words_64(&mut steer_words, slot_w);
                *trail_phase_words_mut(trail_inout) = steer_words;
            }
        }
        Self::copy_trail_and_words_to_slot_fast(
            &mut self.trails[stored_slot_idx as usize],
            trail_inout,
            &steer_words,
        );

        let (steered, subhead_routed, skip_compensated) =
            StigmergicThresholds::execute_decision_steering_and_logit_scaling(
                super::thresholds::DecisionSteerParams {
                    decision,
                    steer_words: &steer_words,
                    phase_step_min_rad,
                    phase_clamp_max,
                    head_dim,
                    num_heads,
                    best_coh,
                },
                query_tensor,
                key_tensor,
                logits_opt,
            )?;

        let rank1_flag = (mode_flags & ACTION_MODE_RANK1_INJECT) != 0;
        let metric = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, LEDGER_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(subhead_routed, LEDGER_METRIC_SUBHEAD_ROUTED, 0)
            | branchless_select_u8(rank1_flag, LEDGER_METRIC_RANK1_INJECTED, 0)
            | branchless_select_u8(skip_compensated, LEDGER_METRIC_LAYER_SKIP_COMPENSATED, 0)
            | branchless_select_u8(contradiction_repulsed, LEDGER_METRIC_REPULSION_APPLIED, 0)
            | branchless_select_u8(found_coherent, LEDGER_METRIC_SLOT_SYNERGY_ACTIVE, 0);
        self.ledger_slot.push_telemetry_metric(metric);

        self.last_selected_trail_id = selected_id;
        self.last_entropy_h_tilde = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_relu_f32(entropy_h_tilde),
            0.0,
        );
        self.last_anomaly_severity = branchless_select_f32(
            contradiction_repulsed,
            0.85,
            branchless_clamp_f32(0.5 * (1.0 - best_coh), 0.0, 1.0),
        );
        self.last_decision = decision as u8;

        Ok(SubstrateStepReport {
            evolved_ptr: evolved_ptr.raw(),
            updated_density: updated_rho,
            phase_coherence: best_coh,
            transition_log_prob: log_prob,
            selected_trail_id: selected_id,
            dominant_anchor_id: trail_inout.target_anchor_id,
            decision: decision as u8,
            action_mode_flags: mode_flags,
            target_slot,
            bloom_repulsion_density: bloom_density,
        })
    }

    /// Resolves the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, even
    /// integer in `2..=head_dim`) for this engine's `head_dim` and last observed entropy $\tilde{H}$.
    #[inline(always)]
    #[must_use]
    pub fn resolve_autonomous_rot_dim(
        &self,
        lineage_flags: u32,
        quorum_mask: u8,
        von_mises_kappa: f32,
    ) -> usize {
        resolve_autonomous_swarm_rot_dim(
            usize::from(self.head_dim),
            lineage_flags,
            quorum_mask,
            self.last_entropy_h_tilde,
            von_mises_kappa,
        )
    }

    /// Traverses a multi-hop **Indirect Pointer Chain** (`start_trail_idx -> linked_trail_id -> ...`)
    /// across the inline L1D Sidecar Pointer Table (`self.trails`), accumulating coherent 64-head
    /// SWAR phase signatures along the pathway, reinforcing traversed memo links, and executing
    /// stigmergic threshold action selection & RoPE steering from the terminal memo.
    ///
    /// Starting from a single entry memo index (`start_trail_idx`), the CPU follows up to
    /// `max_hops` (`1..=SUBSTRATE_INLINE_TRAILS`) linked memos using a 16-bit visited bitmask
    /// for $\mathcal{O}(1)$ zero-allocation cycle prevention. Each coherent hop ($\mathcal{C}_{\mathrm{hop}} > 0$)
    /// contributes its phase delta to `trail_inout` and receives Hebbian/stigmergic pheromone
    /// reinforcement along the traversed pathway.
    ///
    /// Returns `(SubstrateStepReport, hops_traversed, terminal_trail_idx, terminal_anchor_id, mean_chain_coherence)`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `start_trail_idx >= self.inline_active_len()` or
    /// `max_hops == 0`.
    #[inline]
    // v0.3.0 frozen public Rust API; parameter-struct consolidation deferred to v0.4.0.
    #[allow(clippy::too_many_arguments)]
    pub fn traverse_pointer_chain_and_orchestrate(
        &mut self,
        start_trail_idx: usize,
        max_hops: usize,
        entropy_h_tilde: f32,
        trail_inout: &mut PointerTrail,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(SubstrateStepReport, u32, usize, u32, f32)> {
        let active_len = self.inline_active_len();
        if start_trail_idx >= active_len || max_hops == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let bounded_hops = branchless_select_usize(
            max_hops > SUBSTRATE_INLINE_TRAILS,
            SUBSTRATE_INLINE_TRAILS,
            max_hops,
        );

        let mut cur_idx = start_trail_idx;
        let mut visited_mask: u16 = 0;
        let mut hops: u32 = 0;
        let mut coh_sum = 0.0_f32;
        let mut chain_w = *trail_phase_words_ref(&self.trails[start_trail_idx]);
        let mut terminal_ptr = TaggedPtr(self.trails[start_trail_idx].current_ptr);
        let mut terminal_anchor_id = self.trails[start_trail_idx].target_anchor_id;
        let mut max_chain_rho = self.trails[start_trail_idx].stigmergic_density;
        let mut max_chain_pull = self.trails[start_trail_idx].attractor_pull_force;
        let reinf = branchless_relu_f32(self.thresholds.reinforcement_gain) * 0.5;

        let mut step = 0usize;
        while step < bounded_hops {
            let bit = 1u16 << (cur_idx & (SUBSTRATE_INLINE_TRAILS - 1));
            if (visited_mask & bit) != 0 {
                break;
            }
            visited_mask |= bit;
            hops = hops.wrapping_add(1);

            let node_w = *trail_phase_words_ref(&self.trails[cur_idx]);
            let hop_coh = if step == 0 {
                1.0_f32
            } else {
                coherence_from_word_arrays_64(&chain_w, &node_w)
            };

            if hop_coh < -self.thresholds.contradiction_coherence_thresh {
                break;
            }

            if step > 0 {
                swar_or_avx2_q7_add_sat_words_64(&mut chain_w, &node_w);
            }

            coh_sum += hop_coh;
            let node = &mut self.trails[cur_idx];
            let boosted_rho = branchless_min_f32(
                branchless_flush_subnormal_f32(
                    branchless_relu_f32(node.stigmergic_density)
                        + reinf * branchless_relu_f32(hop_coh),
                ),
                MAX_STIGMERGIC_DENSITY,
            );
            node.stigmergic_density = boosted_rho;
            max_chain_rho = branchless_max_f32(max_chain_rho, boosted_rho);
            max_chain_pull = branchless_max_f32(max_chain_pull, node.attractor_pull_force);
            if node.current_ptr != 0 {
                terminal_ptr = TaggedPtr(node.current_ptr);
            }
            if node.target_anchor_id != 0 {
                terminal_anchor_id = node.target_anchor_id;
            }

            let next_id = node.linked_trail_id as usize;
            if next_id >= active_len || next_id == cur_idx {
                break;
            }
            cur_idx = next_id;
            step += 1;
        }

        let mean_coh = branchless_select_f32(
            hops > 0,
            branchless_clamp_f32(coh_sum * fast_inv_len_f32(hops as usize), -1.0, 1.0),
            0.0,
        );
        *trail_phase_words_mut(trail_inout) = chain_w;
        trail_inout.stigmergic_density =
            branchless_max_f32(trail_inout.stigmergic_density, max_chain_rho);
        trail_inout.attractor_pull_force = branchless_min_f32(
            branchless_max_f32(trail_inout.attractor_pull_force, max_chain_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        if terminal_anchor_id != 0 {
            trail_inout.target_anchor_id = terminal_anchor_id;
        }
        trail_inout.linked_trail_id = cur_idx as u32;

        let dom_slot = terminal_ptr.target_attractor_slot();
        let report = self.orchestrate_next_action_step_core(
            terminal_ptr,
            entropy_h_tilde,
            dom_slot,
            trail_inout,
            attention_scores,
            query_tensor,
            key_tensor,
            logits_opt,
            false,
            Some(&chain_w),
            DEFAULT_PHASE_STEP_MIN_RAD,
            DEFAULT_PHASE_CLAMP_MAX_RAD,
        )?;

        Ok((report, hops, cur_idx, terminal_anchor_id, mean_coh))
    }
}

// ===================================================================
// 3. Trait Implementations for `StigmergicSubstrate`
// ===================================================================

impl FastPathLedger for StigmergicSubstrate {
    #[inline]
    fn evaluate_auto_tune(
        &self,
        slot: &LedgerSlot,
        config_out: &mut RuntimeConfig,
    ) -> LmnkResult<()> {
        let word = u64::from_le_bytes(slot.telemetry_mini_ledger);
        let spike_count = (word & 0x8080_8080_8080_8080).count_ones();
        let clamp_max = config_out.phase_clamp_max_rad;
        let base_clamp =
            branchless_select_f32(clamp_max > 0.0, clamp_max, DEFAULT_PHASE_CLAMP_MAX_RAD);
        let damping = 1.0 - 0.025 * (spike_count as f32);
        config_out.head_dim = u32::from(self.head_dim);
        config_out.num_heads = u32::from(self.num_heads);
        config_out.phase_clamp_max_rad = branchless_clamp_f32(base_clamp * damping, 0.01, 0.25);
        config_out.substrate_capacity = self.capacity;
        config_out.ledger_mode = (slot.mode & 0xFF) as u8;
        let _ = LEDGER_HISTORY_LEN;
        Ok(())
    }
}

impl PointerSubstrate for StigmergicSubstrate {
    #[inline]
    fn reset(&mut self) -> LmnkResult<()> {
        let active_len = self.inline_active_len();
        let mut i = 0usize;
        while i < active_len {
            Self::clear_trail_slot_fast(&mut self.trails[i]);
            i += 1;
        }
        self.active_count = 0;
        self.write_cursor = 0;
        self.step_counter = 0;
        self.last_selected_trail_id = 0;
        self.last_entropy_h_tilde = 0.0;
        self.last_anomaly_severity = 0.0;
        self.last_decision = StigmergicActionDecision::PassiveSidecar as u8;
        let _ = self.release_unpinned_lanes();
        Ok(())
    }

    #[inline]
    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> LmnkResult<()> {
        let tp = TaggedPtr(current_ptr);
        self.step_counter = self.step_counter.wrapping_add(1);
        trail_out.record_pointer_transition(current_ptr, 1, self.thresholds.reinforcement_gain);
        let embedded_phase = tp.unpack_phase();
        if embedded_phase != 0 {
            trail_out.accumulate_uniform_phase(embedded_phase);
        }
        trail_out.cluster_id = branchless_select_u32(
            trail_out.target_anchor_id != 0,
            trail_out.target_anchor_id,
            u32::from(tp.target_attractor_slot() & ATTRACTOR_SLOT_ID_MASK),
        );
        trail_out.sync_flags_with_current_ptr();

        let slot_idx = (self.write_cursor as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        Self::copy_trail_to_slot_fast(&mut self.trails[slot_idx], trail_out);
        self.write_cursor =
            self.write_cursor.wrapping_add(1) & ((SUBSTRATE_INLINE_TRAILS as u32) - 1);
        let cap = self.capacity;
        self.active_count = branchless_select_u32(
            self.active_count < cap,
            self.active_count.wrapping_add(1),
            self.active_count,
        );
        Ok(())
    }

    #[inline]
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: PhaseQ7,
    ) -> LmnkResult<()> {
        if trail_id >= self.capacity {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let inline_idx = (trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let min_active = ((inline_idx as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
        self.trails[inline_idx].accumulate_uniform_phase(phase_delta);
        let slot_idx = (trail_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        if phase_delta != 0 {
            let splat = ((phase_delta as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
            // SAFETY: `self.slot_phase_bank` is 64B-aligned and `slot_idx < 8`.
            let w = unsafe {
                &mut *(self
                    .slot_phase_bank
                    .as_mut_slice()
                    .as_mut_ptr()
                    .add(slot_idx * MAX_ATTENTION_HEADS)
                    as *mut [u64; NUM_STIGMERGIC_GRANULES])
            };
            swar_or_avx2_q7_add_sat_words_64(w, &[splat; NUM_STIGMERGIC_GRANULES]);
            self.slot_enabled_mask |= 1u8 << slot_idx;
        }
        Ok(())
    }

    #[inline]
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> LmnkResult<()> {
        if trail_id >= self.capacity || !branchless_is_finite_f32(pull_force) {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let clean_pull =
            branchless_min_f32(branchless_relu_f32(pull_force), MAX_ATTRACTOR_PULL_FORCE);
        let inline_idx = (trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let min_active = ((inline_idx as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
        let trail = &mut self.trails[inline_idx];
        trail.target_anchor_id = anchor_id;
        trail.cluster_id = branchless_select_u32(anchor_id != 0, anchor_id, trail.cluster_id);
        trail.attractor_pull_force = clean_pull;
        let slot_idx = (anchor_id as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        self.slot_weights[slot_idx] = clean_pull;
        Ok(())
    }

    #[inline]
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> LmnkResult<()> {
        if src_trail_id >= self.capacity || dst_trail_id >= self.capacity {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let src_idx = (src_trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let dst_idx = (dst_trail_id as usize) & (SUBSTRATE_INLINE_TRAILS - 1);
        let min_active = ((src_idx.max(dst_idx) as u32) + 1).min(self.capacity);
        self.active_count = branchless_select_u32(
            self.active_count < min_active,
            min_active,
            self.active_count,
        );
        self.trails[src_idx].linked_trail_id = dst_trail_id;
        self.last_selected_trail_id = dst_trail_id;
        Ok(())
    }

    #[inline(always)]
    fn capacity(&self) -> u32 {
        self.capacity
    }

    #[inline(always)]
    fn active_count(&self) -> u32 {
        self.active_count
    }

    #[inline(always)]
    fn lane_mask(&self) -> u8 {
        self.slot_enabled_mask
    }

    #[inline(always)]
    fn repulsion_lane_mask(&self) -> u8 {
        self.slot_repulsion_mask
    }

    #[inline(always)]
    fn sandbox_lane_mask(&self) -> u8 {
        self.slot_sandbox_mask
    }

    #[inline(always)]
    fn pinned_lane_mask(&self) -> u8 {
        self.slot_pinned_mask
    }

    #[inline]
    fn lane_words_and_weight(&self, lane: u8) -> Option<([u64; NUM_STIGMERGIC_GRANULES], f32)> {
        let idx = usize::from(lane);
        if idx >= MAX_ATTRACTOR_DOMAINS || ((self.slot_enabled_mask >> idx) & 1) == 0 {
            return None;
        }
        let start = idx * MAX_ATTENTION_HEADS;
        let end = start + MAX_ATTENTION_HEADS;
        let mut arr = [0_i8; MAX_ATTENTION_HEADS];
        arr.copy_from_slice(&self.slot_phase_bank.data[start..end]);
        let words = load_q7_words_64(&arr);
        if swar_words_nonzero_64(&words) {
            Some((words, self.slot_weights[idx]))
        } else {
            None
        }
    }
}

impl StigmergicSubstrate {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    #[inline]
    pub fn step_chunk(&mut self, _mask: &PagedBitmask, chunk_len: u32) -> LmnkResult<()> {
        if chunk_len == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        self.step_counter = self.step_counter.wrapping_add(u64::from(chunk_len));
        Ok(())
    }
}
