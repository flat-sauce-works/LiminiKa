// name: src/core/src/runtime/gateway/frame/synthesis.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-origin [`GatewayFrame`] synthesis and pair-aperture blending.

use super::super::*;
use super::signal::{
    coherence_for_active_heads_64, shape_granule_word_by_aperture,
    swar_tangent_lerp_words_with_uniform_flag_64, tangent_cosine_active_heads_64,
};
use super::{GatewayFrame, ManifoldGatewaySignal};
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_is_finite_f32,
    branchless_lerp_f32, branchless_max_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_rad_to_phase_step_u8,
    branchless_relu_f32, coherence_from_uniform_words_x8,
    coherence_from_word_arrays_with_both_uniform_flags_64, fast_inv_pos_normal_f32,
    is_word_array_uniform_64, pack_action_slot_tag, swar_or_avx2_q7_add_sat_words_64,
    swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_mean_i8_from_words_64,
    swar_q7_mean_i8_from_words_with_uniform_flag_64, swar_q7_neg_sat_x8,
    swar_q7_phase_step_filter_words_with_uniform_flag_64, swar_q7_scale_le128_with_bias_x8,
    swar_slerp_words_with_uniform_flag_64, swar_words_equal_64, swar_words_nonzero_64,
    words_to_q7_array_64, TaggedPtr, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_STEER,
    ACTION_MODE_RANK1_INJECT, ACTION_MODE_SUBHEAD_ROUTE, ACTION_PRESET_EXPRESS_PHASE,
    ACTION_PRESET_INERTIAL_SKIP, ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK,
    DEFAULT_PHASE_CLAMP_MAX_RAD, DEFAULT_PHASE_STEP_MIN_RAD, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u64, branchless_select_u8,
    PhaseQ7, DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
};
use crate::runtime::context::CoreStateStack;
use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, ATTRACTOR_SLOT_REPULSION,
    ATTRACTOR_SLOT_SYSTEM_BASELINE, DEFAULT_FACT_PHASE_CLAMP_RAD, DEFAULT_PERSONA_PHASE_CLAMP_RAD,
    MAX_ATTRACTOR_PULL_FORCE,
};

/// Applies saturating anti-phase repulsion to `blended_w` in-place and returns whether the
/// resulting 64-head phase array is uniform across all 8 SWAR granules.
///
/// When `guard_w` carries a non-zero repulsive pole $R$, computes its saturating anti-phase $-R$
/// (`i8::MIN -> +127`) and superposes it onto `blended_w` via saturating Q7 addition ($A + (-R)$),
/// preserving orthogonal attractive heads without inversion while emitting $-R$ on the repulsive
/// pole's heads (and exact $-R$ when `blended_w` is zero for a solo repulsive pole). When
/// `guard_w` is all zeros (phase-less candidate detector such as a Bloom filter or scalar
/// obstruction guard), falls back to saturating negation of `blended_w` ($-A$).
#[inline]
fn apply_repulsion_pole_words_64(
    blended_w: &mut [u64; NUM_STIGMERGIC_GRANULES],
    blended_uni: bool,
    guard_w: &[u64; NUM_STIGMERGIC_GRANULES],
) -> bool {
    if swar_words_nonzero_64(guard_w) {
        let anti_guard_w = [
            swar_q7_neg_sat_x8(guard_w[0]),
            swar_q7_neg_sat_x8(guard_w[1]),
            swar_q7_neg_sat_x8(guard_w[2]),
            swar_q7_neg_sat_x8(guard_w[3]),
            swar_q7_neg_sat_x8(guard_w[4]),
            swar_q7_neg_sat_x8(guard_w[5]),
            swar_q7_neg_sat_x8(guard_w[6]),
            swar_q7_neg_sat_x8(guard_w[7]),
        ];
        swar_or_avx2_q7_add_sat_words_64(blended_w, &anti_guard_w);
        is_word_array_uniform_64(blended_w)
    } else if blended_uni {
        let neg0 = swar_q7_neg_sat_x8(blended_w[0]);
        *blended_w = [neg0; NUM_STIGMERGIC_GRANULES];
        true
    } else {
        blended_w[0] = swar_q7_neg_sat_x8(blended_w[0]);
        blended_w[1] = swar_q7_neg_sat_x8(blended_w[1]);
        blended_w[2] = swar_q7_neg_sat_x8(blended_w[2]);
        blended_w[3] = swar_q7_neg_sat_x8(blended_w[3]);
        blended_w[4] = swar_q7_neg_sat_x8(blended_w[4]);
        blended_w[5] = swar_q7_neg_sat_x8(blended_w[5]);
        blended_w[6] = swar_q7_neg_sat_x8(blended_w[6]);
        blended_w[7] = swar_q7_neg_sat_x8(blended_w[7]);
        false
    }
}

#[inline]
fn combine_pair_aperture_from_frames(
    fa: GatewayFrame,
    wa: [u64; NUM_STIGMERGIC_GRANULES],
    (mask_a, gain_a, shift_a): (u8, u8, i8),
    fb: GatewayFrame,
    wb: [u64; NUM_STIGMERGIC_GRANULES],
    (mask_b, gain_b, shift_b): (u8, u8, i8),
    stack: &CoreStateStack,
) -> GatewayFrame {
    let ea = branchless_flush_subnormal_f32(branchless_relu_f32(fa.effective_energy));
    let eb = branchless_flush_subnormal_f32(branchless_relu_f32(fb.effective_energy));
    let tot_e = ea + eb;
    let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
    let ratio_b = branchless_clamp_f32(eb * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
    let s_u64 = ((ratio_b * 128.0 + 0.5) as u8).min(128) as u64;
    let inv_s = 128u64.wrapping_sub(s_u64);
    let bias_a = s_u64.wrapping_mul(SWAR_SPLAT_BYTE_64);
    let bias_b = inv_s.wrapping_mul(SWAR_SPLAT_BYTE_64);

    let ego_active = stack.ego.is_active();
    let mut out_words = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let bit = 1u8 << g;
        let in_a = (mask_a & bit) != 0;
        let in_b = (mask_b & bit) != 0;

        let aw = shape_granule_word_by_aperture(wa[g], gain_a, shift_a);
        let bw = shape_granule_word_by_aperture(wb[g], gain_b, shift_b);

        let coh = coherence_from_uniform_words_x8(aw, bw);
        let conflict = coh < -0.25;
        let blended = swar_q7_circular_add_x8(
            swar_q7_scale_le128_with_bias_x8(aw, inv_s, bias_a),
            swar_q7_scale_le128_with_bias_x8(bw, s_u64, bias_b),
        );
        let ego_phase_word =
            (stack.ego.sovereign_granule_phase_q7[g] as u8 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let coh_ea = coherence_from_uniform_words_x8(aw, ego_phase_word);
        let coh_eb = coherence_from_uniform_words_x8(bw, ego_phase_word);
        let prefer_b_tie =
            branchless_select_u8(ego_active, u8::from(coh_eb > coh_ea), u8::from(eb > ea)) != 0;
        let conflict_choice = branchless_select_u64(prefer_b_tie, bw, aw);
        let overlap_w = branchless_select_u64(conflict, conflict_choice, blended);

        let mut gw = branchless_select_u64(in_a & !in_b, aw, 0);
        gw = branchless_select_u64(!in_a & in_b, bw, gw);
        gw = branchless_select_u64(in_a & in_b, overlap_w, gw);
        out_words[g] = gw;
        g += 1;
    }

    let active_heads = fa.active_head_count.min(fb.active_head_count).max(1);
    let pair_coh = if active_heads < MAX_ATTENTION_HEADS as u16 {
        tangent_cosine_active_heads_64(&wa, &wb, active_heads)
    } else {
        coherence_from_word_arrays_with_both_uniform_flags_64(
            &wa,
            is_word_array_uniform_64(&wa),
            &wb,
            is_word_array_uniform_64(&wb),
        )
    };
    let dom_a = ea >= eb;
    let combined_mask = mask_a | mask_b;
    let overlap_mask = mask_a & mask_b;
    let coop_bonus = 1.0 + 0.04 * (overlap_mask.count_ones() as f32);
    let eff_energy = branchless_min_f32(
        branchless_select_f32(
            combined_mask != 0,
            (ea + eb) * 0.5 * coop_bonus + 0.5 * branchless_max_f32(ea, eb),
            0.0,
        ),
        MAX_ATTRACTOR_PULL_FORCE,
    );
    let mut combined = GatewayFrame {
        tagged_ptr: 0,
        mapped_section_offset48: branchless_select_u64(
            dom_a & (fa.mapped_section_offset48 != 0),
            fa.mapped_section_offset48,
            branchless_select_u64(
                fb.mapped_section_offset48 != 0,
                fb.mapped_section_offset48,
                fa.mapped_section_offset48,
            ),
        ),
        target_anchor_id: branchless_select_u32(dom_a, fa.target_anchor_id, fb.target_anchor_id),
        effective_energy: eff_energy,
        phase_clamp_rad: branchless_lerp_f32(fa.phase_clamp_rad, fb.phase_clamp_rad, ratio_b),
        phase_step_min_rad: branchless_min_f32(fa.phase_step_min_rad, fb.phase_step_min_rad),
        effective_inv_temp: branchless_lerp_f32(
            fa.effective_inv_temp,
            fb.effective_inv_temp,
            ratio_b,
        ),
        fact_persona_coherence: pair_coh,
        repulsion_severity: branchless_max_f32(fa.repulsion_severity, fb.repulsion_severity),
        lineage_flags: fa.lineage_flags | fb.lineage_flags,
        attribution_tag: branchless_select_u8(
            fa.attribution_tag >= fb.attribution_tag,
            fa.attribution_tag,
            fb.attribution_tag,
        ),
        dominant_slot: branchless_select_u8(dom_a, fa.dominant_slot, fb.dominant_slot),
        action_mode_flags: fa.action_mode_flags | fb.action_mode_flags,
        active_swarm_cell_mask: combined_mask
            & (fa.active_swarm_cell_mask | fb.active_swarm_cell_mask),
        phase_step_min_q7: branchless_select_u8(
            fa.phase_step_min_q7 <= fb.phase_step_min_q7,
            fa.phase_step_min_q7,
            fb.phase_step_min_q7,
        ),
        subhead_fact_heads: branchless_select_u8(
            dom_a,
            fa.subhead_fact_heads,
            fb.subhead_fact_heads,
        ),
        subhead_logic_heads: branchless_select_u8(
            dom_a,
            fa.subhead_logic_heads,
            fb.subhead_logic_heads,
        ),
        subhead_explore_heads: branchless_select_u8(
            dom_a,
            fa.subhead_explore_heads,
            fb.subhead_explore_heads,
        ),
        subhead_fact_gain_q7: branchless_select_u8(
            dom_a,
            fa.subhead_fact_gain_q7,
            fb.subhead_fact_gain_q7,
        ),
        subhead_logic_gain_q7: branchless_select_u8(
            dom_a,
            fa.subhead_logic_gain_q7,
            fb.subhead_logic_gain_q7,
        ),
        subhead_explore_gain_q7: branchless_select_u8(
            dom_a,
            fa.subhead_explore_gain_q7,
            fb.subhead_explore_gain_q7,
        ),
        mean_phase_tag_q7: 0,
        active_head_count: active_heads,
        frame_flags: 0,
        phase_deltas: words_to_q7_array_64(&out_words),
    };
    let tag_q7 = swar_q7_mean_i8_from_words_64(combined.phase_words_ref());
    let ptr_tag = pack_action_slot_tag(
        combined.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
        combined.action_mode_flags,
    );
    combined.mean_phase_tag_q7 = tag_q7;
    combined.frame_flags = ptr_tag as u16;
    combined.tagged_ptr = TaggedPtr::pack(combined.mapped_section_offset48, ptr_tag, tag_q7).raw();
    combined
}

#[inline]
pub(super) fn combine_pair_aperture_frame_pure<
    A: ManifoldGatewaySignal + ?Sized,
    B: ManifoldGatewaySignal + ?Sized,
>(
    a: &A,
    b: &B,
    stack: &CoreStateStack,
) -> GatewayFrame {
    let ap_a = a.resolve_dynamic_core_aperture(stack);
    let ap_b = b.resolve_dynamic_core_aperture(stack);
    let (fa, wa) = a.synthesize_gateway_frame_with_words();
    let (fb, wb) = b.synthesize_gateway_frame_with_words();
    combine_pair_aperture_from_frames(fa, wa, ap_a, fb, wb, ap_b, stack)
}

#[inline]
pub(super) fn combine_fact_persona_aperture_frame_pure<
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
>(
    fact: &F,
    persona: &P,
    stack: &CoreStateStack,
) -> GatewayFrame {
    let ap_a = fact.resolve_dynamic_core_aperture(stack);
    let ap_b = persona.resolve_dynamic_core_aperture(stack);
    let (fa, wa) = synthesize_multi_origin_gateway_frame_with_words::<
        SystemConstraint<()>,
        F,
        SubjectivePersona<[i8; 64]>,
        ContradictionGuard<()>,
    >(None, Some(fact), None, None);
    let (fb, wb) = synthesize_multi_origin_gateway_frame_with_words::<
        SystemConstraint<()>,
        ObjectiveFact<[i8; 64]>,
        P,
        ContradictionGuard<()>,
    >(None, None, Some(persona), None);
    combine_pair_aperture_from_frames(fa, wa, ap_a, fb, wb, ap_b, stack)
}

#[inline]
pub(super) fn apply_sys_and_guard_to_pair_frame<
    S: SystemConstraintSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
>(
    sys: &S,
    guard_opt: Option<&G>,
    mut frame: GatewayFrame,
) -> GatewayFrame {
    let mut prov_bits =
        frame.lineage_flags | sys.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT;
    let mut active_swarm_mask = frame.active_swarm_cell_mask & sys.active_swarm_cells_mask();
    let force_skip = sys.recommend_layer_skip();
    let min_step_rad = branchless_max_f32(0.0, sys.phase_step_min_rad_override());
    let sys_sev = branchless_clamp_f32(sys.causal_anomaly_severity(), 0.0, 1.0);
    if force_skip | (sys.active_swarm_cells_mask() != 0xFF) {
        prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
    }

    let mut rep_sev = branchless_max_f32(frame.repulsion_severity, sys_sev);
    let words = *frame.phase_words_ref();
    if let Some(guard) = guard_opt {
        const ZERO_PHASES_64: [PhaseQ7; MAX_ATTENTION_HEADS] = [0i8; MAX_ATTENTION_HEADS];
        prov_bits |= guard.causal_provenance_bits() | CAUSAL_AXIS_CONTRADICTION_GUARD;
        let g_sev = if guard.requires_phase_slice_for_repulsion() {
            guard.evaluate_repulsion_severity_from_words(&words, &frame.phase_deltas)
        } else {
            guard.evaluate_repulsion_severity_from_words(&words, &ZERO_PHASES_64)
        };
        rep_sev = branchless_clamp_f32(
            branchless_max_f32(
                rep_sev,
                branchless_max_f32(g_sev, guard.causal_anomaly_severity()),
            ),
            0.0,
            1.0,
        );
    }

    let repulsion_triggered =
        (rep_sev > 0.50) | ((prov_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
    if repulsion_triggered {
        prov_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
        let (guard_w, guard_e) = match guard_opt {
            Some(guard) => (
                guard.project_repulsive_pole_words_64(),
                branchless_flush_subnormal_f32(branchless_relu_f32(guard.grounding_base_energy())),
            ),
            None => ([0u64; NUM_STIGMERGIC_GRANULES], 0.0_f32),
        };
        let mut out_w = words;
        let _ = apply_repulsion_pole_words_64(&mut out_w, false, &guard_w);
        frame.phase_deltas = words_to_q7_array_64(&out_w);
        frame.dominant_slot = ATTRACTOR_SLOT_REPULSION;
        rep_sev = branchless_max_f32(rep_sev, 0.85);
        if frame.effective_energy <= DEFAULT_ENTROPY_EPS {
            frame.effective_energy = branchless_min_f32(
                branchless_max_f32(guard_e, rep_sev),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }
        if active_swarm_mask == 0 {
            active_swarm_mask = sys.active_swarm_cells_mask();
        }
        frame.action_mode_flags = ACTION_PRESET_REPULSION_ESCAPE;
        frame.attribution_tag = ROOT_CAUSE_COHOMOLOGY_REPULSION;
        frame.effective_inv_temp = 0.75;
    } else if force_skip {
        frame.action_mode_flags |= ACTION_MODE_LAYER_SKIP;
    }

    frame.lineage_flags = prov_bits;
    frame.active_swarm_cell_mask = active_swarm_mask;
    frame.phase_step_min_rad = branchless_max_f32(frame.phase_step_min_rad, min_step_rad);
    frame.phase_step_min_q7 = branchless_rad_to_phase_step_u8(frame.phase_step_min_rad);
    frame.repulsion_severity = rep_sev;
    let tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
    let ptr_tag = pack_action_slot_tag(
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
        frame.action_mode_flags,
    );
    frame.mean_phase_tag_q7 = tag_q7;
    frame.frame_flags = ptr_tag as u16;
    frame.tagged_ptr = TaggedPtr::pack(frame.mapped_section_offset48, ptr_tag, tag_q7).raw();
    frame
}

impl CausalProvenanceSignal for GatewayFrame {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.lineage_flags
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.repulsion_severity
    }
}

impl ManifoldGatewaySignal for GatewayFrame {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        *self
    }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (*self, *self.phase_words_ref())
    }
}

/// Synthesizes both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64` SWAR phase words
/// (`[u64; 8]`) from optional **Origin 1 (`SystemConstraintSignal`)** and **Origin 2
/// (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame_with_words<S, F, P, G>(
    sys_opt: Option<&S>,
    fact_opt: Option<&F>,
    persona_opt: Option<&P>,
    guard_opt: Option<&G>,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])
where
    S: SystemConstraintSignal + ?Sized,
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
{
    let mut prov_bits = 0u32;
    let mut active_swarm_mask = 0xFFu8;
    let mut force_skip = false;
    let mut min_step_rad = DEFAULT_PHASE_STEP_MIN_RAD;
    let mut sys_sev = 0.0_f32;

    if let Some(sys) = sys_opt {
        prov_bits |= sys.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT;
        active_swarm_mask = sys.active_swarm_cells_mask();
        force_skip = sys.recommend_layer_skip();
        min_step_rad = branchless_max_f32(0.0, sys.phase_step_min_rad_override());
        sys_sev = branchless_clamp_f32(sys.causal_anomaly_severity(), 0.0, 1.0);
        if force_skip | (active_swarm_mask != 0xFF) {
            prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
        }
    }
    let min_step_q7 = branchless_rad_to_phase_step_u8(min_step_rad);

    let mut has_fact = false;
    let mut fact_e = 0.0_f32;
    let mut fact_addr = 0u64;
    let mut fact_id = 0u32;
    let mut fact_clamp = DEFAULT_FACT_PHASE_CLAMP_RAD;
    let mut fact_gain_q7 = 128u8;
    let mut fact_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut fact_heads = MAX_ATTENTION_HEADS as u16;

    if let Some(fact) = fact_opt {
        has_fact = true;
        prov_bits |= fact.causal_provenance_bits() | CAUSAL_AXIS_OBJECTIVE_FACT;
        let (fw, fe, fa, fid, fh) = fact.synthesize_grounding_envelope();
        fact_w = fw;
        fact_e = branchless_flush_subnormal_f32(branchless_relu_f32(fe));
        fact_addr = fa & TaggedPtr::ADDR_MASK;
        fact_id = fid;
        fact_heads = fh.clamp(1, MAX_ATTENTION_HEADS as u16);
        let fc = fact.fact_phase_clamp_rad();
        fact_clamp = branchless_select_f32(
            branchless_is_finite_f32(fc) & (fc > 0.0),
            fc,
            DEFAULT_FACT_PHASE_CLAMP_RAD,
        );
        let fg = branchless_clamp_f32(fact.fact_subhead_gain(), 0.25, 1.99);
        fact_gain_q7 = (fg * 128.0 + 0.5) as u8;
        if fact_e < 0.25 {
            prov_bits |= CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK;
        }
    }

    let mut has_persona = false;
    let mut pers_e = 0.0_f32;
    let mut pers_addr = 0u64;
    let mut pers_id = 0u32;
    let mut pers_clamp = DEFAULT_PERSONA_PHASE_CLAMP_RAD;
    let mut pers_gain_q7 = 128u8;
    let mut pers_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut pers_heads = MAX_ATTENTION_HEADS as u16;

    if let Some(pers) = persona_opt {
        has_persona = true;
        prov_bits |= pers.causal_provenance_bits() | CAUSAL_AXIS_SUBJECTIVE_PERSONA;
        let (pw, pe, pa, pid, ph) = pers.synthesize_grounding_envelope();
        pers_w = pw;
        pers_e = branchless_flush_subnormal_f32(branchless_relu_f32(pe));
        pers_addr = pa & TaggedPtr::ADDR_MASK;
        pers_id = pid;
        pers_heads = ph.clamp(1, MAX_ATTENTION_HEADS as u16);
        let pc = pers.persona_phase_clamp_rad();
        pers_clamp = branchless_select_f32(
            branchless_is_finite_f32(pc) & (pc > 0.0),
            pc,
            DEFAULT_PERSONA_PHASE_CLAMP_RAD,
        );
        let pg = branchless_clamp_f32(pers.persona_subhead_gain(), 0.25, 1.99);
        pers_gain_q7 = (pg * 128.0 + 0.5) as u8;
    }

    let active_heads = if has_fact & has_persona {
        fact_heads.min(pers_heads)
    } else if has_fact {
        fact_heads
    } else if has_persona {
        pers_heads
    } else {
        MAX_ATTENTION_HEADS as u16
    };

    let mut fp_coh = 1.0_f32;
    let mut collision = false;
    let mut blended_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut blended_uni = true;
    let mut avx_precomputed_tag: Option<PhaseQ7> = None;
    let mut eff_energy = 0.0_f32;
    let mut raw_clamp = DEFAULT_PERSONA_PHASE_CLAMP_RAD;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut dom_anchor_id = 0u32;
    let mut storage_addr = 0u64;
    let mut subhead_fact = 24u8;
    let mut subhead_logic = 24u8;
    let mut subhead_explore = 16u8;

    if has_fact & has_persona {
        let fact_uni = is_word_array_uniform_64(&fact_w);
        let pers_uni = is_word_array_uniform_64(&pers_w);
        let both_uni = fact_uni & pers_uni;
        let same_fp = if both_uni {
            fact_w[0] == pers_w[0]
        } else {
            swar_words_equal_64(&fact_w, &pers_w)
        };
        let diff_w0 = if both_uni {
            swar_q7_circular_diff_x8(pers_w[0], fact_w[0])
        } else {
            0
        };
        fp_coh = if same_fp {
            1.0
        } else if both_uni {
            coherence_from_uniform_words_x8(pers_w[0], fact_w[0])
        } else if active_heads < MAX_ATTENTION_HEADS as u16 {
            tangent_cosine_active_heads_64(&pers_w, &fact_w, active_heads)
        } else {
            coherence_for_active_heads_64(&pers_w, pers_uni, &fact_w, fact_uni, active_heads)
        };
        collision = fp_coh < -DEFAULT_PERSONA_FACT_COLLISION_THRESH;
        if collision {
            prov_bits |= CAUSAL_CAUSE_PERSONA_FACT_COLLISION;
        }
        // When operating in a sub-64 active-head tangent subspace (`active_heads < 64`) and
        // ObjectiveFact dominates (`fact_e >= pers_e`), gate Persona phase blending by strong
        // phase-lock coherence (`fp_coh > 0.70`) so an unaligned Persona slice never dilutes
        // factual steering; on full 64-head toroidal signals preserve standard S^1 SLERP synergy.
        let is_subspace = active_heads < MAX_ATTENTION_HEADS as u16;
        let coh_gate = if is_subspace {
            branchless_select_f32(
                fact_e >= pers_e,
                branchless_relu_f32((fp_coh - 0.70) * 3.333_333_3),
                branchless_clamp_f32(1.0 + 0.5 * fp_coh, 0.5, 1.5),
            )
        } else {
            branchless_clamp_f32(1.0 + 0.5 * fp_coh, 0.5, 1.5)
        };
        let eff_pers_e = branchless_select_f32(
            collision,
            if is_subspace { 0.0 } else { pers_e * 0.20 },
            pers_e * coh_gate,
        );
        let tot_e = fact_e + eff_pers_e;
        let safe_tot = branchless_select_f32(tot_e > DEFAULT_ENTROPY_EPS, tot_e, 1.0);
        let pers_ratio =
            branchless_clamp_f32(eff_pers_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
        let s_u64 = ((pers_ratio * 128.0 + 0.5) as u8).min(128) as u64;
        if same_fp | (s_u64 == 0) {
            blended_w = fact_w;
            blended_uni = fact_uni;
        } else if s_u64 >= 128 {
            blended_w = pers_w;
            blended_uni = pers_uni;
        } else if both_uni {
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let step0 = swar_q7_scale_le128_with_bias_x8(diff_w0, s_u64, bias_restore);
            blended_w = [swar_q7_circular_add_x8(fact_w[0], step0); NUM_STIGMERGIC_GRANULES];
            blended_uni = true;
        } else if is_subspace {
            let (bw, bu) = swar_tangent_lerp_words_with_uniform_flag_64(
                &fact_w, fact_uni, &pers_w, pers_uni, s_u64,
            );
            blended_w = bw;
            blended_uni = bu;
        } else {
            let bias_restore = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let (bw, bu) = swar_slerp_words_with_uniform_flag_64(
                &fact_w,
                fact_uni,
                &pers_w,
                pers_uni,
                s_u64,
                bias_restore,
            );
            blended_w = bw;
            blended_uni = bu;
        }
        let synergy_boost =
            branchless_select_f32(collision, 1.0, 1.0 + 0.25 * branchless_relu_f32(fp_coh));
        eff_energy = branchless_min_f32(tot_e * synergy_boost, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = branchless_lerp_f32(fact_clamp, pers_clamp, pers_ratio);

        let fact_dom = (fact_e >= pers_e) | collision;
        dom_slot = branchless_select_u8(fact_dom, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA);
        dom_anchor_id = branchless_select_u32(
            fact_dom & (fact_id != 0),
            fact_id,
            branchless_select_u32(pers_id != 0, pers_id, fact_id),
        );
        storage_addr = branchless_select_u64(
            fact_addr != 0,
            fact_addr,
            branchless_select_u64(pers_addr != 0, pers_addr, 0),
        );
        subhead_fact = branchless_select_u8(fact_dom, 32, 20);
        subhead_logic = 24;
        subhead_explore = branchless_select_u8(fact_dom, 8, 20);
    } else if has_fact {
        blended_w = fact_w;
        blended_uni = is_word_array_uniform_64(&fact_w);
        eff_energy = branchless_min_f32(fact_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = fact_clamp;
        dom_slot = ATTRACTOR_SLOT_KNOWLEDGE;
        dom_anchor_id = fact_id;
        storage_addr = fact_addr;
        subhead_fact = 36;
        subhead_logic = 20;
        subhead_explore = 8;
    } else if has_persona {
        blended_w = pers_w;
        blended_uni = is_word_array_uniform_64(&pers_w);
        eff_energy = branchless_min_f32(pers_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = pers_clamp;
        dom_slot = ATTRACTOR_SLOT_PERSONA;
        dom_anchor_id = pers_id;
        storage_addr = pers_addr;
        subhead_fact = 16;
        subhead_logic = 24;
        subhead_explore = 24;
    }

    let norm_phase_clamp = {
        if raw_clamp < DEFAULT_FACT_PHASE_CLAMP_RAD {
            let auth_ratio = branchless_clamp_f32(raw_clamp * INV_FACT_PHASE_CLAMP_RAD, 0.10, 1.0);
            let auth_u64 = ((auth_ratio * 128.0 + 0.5) as u8).min(128) as u64;
            if auth_u64 < 128 {
                let bias_auth = (128u64.wrapping_sub(auth_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                if blended_uni {
                    blended_w =
                        [swar_q7_scale_le128_with_bias_x8(blended_w[0], auth_u64, bias_auth);
                            NUM_STIGMERGIC_GRANULES];
                } else {
                    blended_w = [
                        swar_q7_scale_le128_with_bias_x8(blended_w[0], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[1], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[2], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[3], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[4], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[5], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[6], auth_u64, bias_auth),
                        swar_q7_scale_le128_with_bias_x8(blended_w[7], auth_u64, bias_auth),
                    ];
                }
            }
        }
        DEFAULT_PHASE_CLAMP_MAX_RAD
    };

    if min_step_q7 > 1 {
        let (fw, fu) = swar_q7_phase_step_filter_words_with_uniform_flag_64(
            &blended_w,
            blended_uni,
            min_step_q7,
        );
        blended_w = fw;
        blended_uni = fu;
    }

    let mut rep_sev = 0.0_f32;
    if let Some(guard) = guard_opt {
        const ZERO_PHASES_64: [PhaseQ7; MAX_ATTENTION_HEADS] = [0i8; MAX_ATTENTION_HEADS];
        prov_bits |= guard.causal_provenance_bits() | CAUSAL_AXIS_CONTRADICTION_GUARD;
        let g_sev = if guard.requires_phase_slice_for_repulsion() {
            let guard_phases = words_to_q7_array_64(&blended_w);
            guard.evaluate_repulsion_severity_from_words(&blended_w, &guard_phases)
        } else {
            guard.evaluate_repulsion_severity_from_words(&blended_w, &ZERO_PHASES_64)
        };
        rep_sev = branchless_clamp_f32(
            branchless_max_f32(g_sev, guard.causal_anomaly_severity()),
            0.0,
            1.0,
        );
    }
    let repulsion_triggered =
        (rep_sev > 0.50) | ((prov_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0);
    if repulsion_triggered {
        prov_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
        avx_precomputed_tag = None;
        let (guard_w, guard_e) = match guard_opt {
            Some(guard) => (
                guard.project_repulsive_pole_words_64(),
                branchless_flush_subnormal_f32(branchless_relu_f32(guard.grounding_base_energy())),
            ),
            None => ([0u64; NUM_STIGMERGIC_GRANULES], 0.0_f32),
        };
        blended_uni = apply_repulsion_pole_words_64(&mut blended_w, blended_uni, &guard_w);
        dom_slot = ATTRACTOR_SLOT_REPULSION;
        rep_sev = branchless_max_f32(rep_sev, 0.85);
        if !(has_fact | has_persona) || eff_energy <= DEFAULT_ENTROPY_EPS {
            eff_energy = branchless_min_f32(
                branchless_max_f32(guard_e, rep_sev),
                MAX_ATTRACTOR_PULL_FORCE,
            );
        }
    }
    let phase_deltas = words_to_q7_array_64(&blended_w);

    let mut action_flags = if repulsion_triggered {
        ACTION_PRESET_REPULSION_ESCAPE
    } else if force_skip {
        ACTION_PRESET_INERTIAL_SKIP
    } else if has_fact | has_persona {
        ACTION_MODE_PHASE_STEER
            | ACTION_MODE_SUBHEAD_ROUTE
            | branchless_select_u8(eff_energy >= 2.0, ACTION_MODE_RANK1_INJECT, 0)
    } else {
        ACTION_PRESET_EXPRESS_PHASE
    };
    if force_skip & (!repulsion_triggered) {
        action_flags |= ACTION_MODE_LAYER_SKIP;
    }

    let dominant_cause = if repulsion_triggered {
        ROOT_CAUSE_COHOMOLOGY_REPULSION
    } else if collision {
        ROOT_CAUSE_PERSONA_FACT_COLLISION
    } else if (prov_bits & CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK) != 0 {
        ROOT_CAUSE_FACT_RETRIEVAL_MISS
    } else if (prov_bits & CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE) != 0 {
        ROOT_CAUSE_SYSTEM_QOS_THROTTLE
    } else {
        ROOT_CAUSE_NOMINAL
    };

    let inv_temp = if repulsion_triggered {
        0.75
    } else if has_fact | has_persona {
        branchless_phase_coherence_inv_temp_scale(fp_coh, 0.65)
    } else {
        1.0
    };

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if !blended_uni && avx_precomputed_tag.is_none() {
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        let (v0, v1) = unsafe { crate::foundation::phase_ops::avx2_words64_to_vec64(&blended_w) };
        // SAFETY: `target_feature = "avx2"` is enabled at compile time.
        avx_precomputed_tag =
            Some(unsafe { crate::foundation::avx2_q7_mean_i8_from_vec64(v0, v1) });
    }
    let tag_q7 = match avx_precomputed_tag {
        Some(t) => t,
        None => swar_q7_mean_i8_from_words_with_uniform_flag_64(&blended_w, blended_uni),
    };
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = TaggedPtr::pack(storage_addr, packed_tag, tag_q7).raw();
    let logic_gain_q7 = ((fact_gain_q7 as u16 + pers_gain_q7 as u16) >> 1) as u8;

    (
        GatewayFrame {
            tagged_ptr,
            mapped_section_offset48: storage_addr,
            target_anchor_id: dom_anchor_id,
            effective_energy: eff_energy,
            phase_clamp_rad: norm_phase_clamp,
            phase_step_min_rad: min_step_rad,
            effective_inv_temp: inv_temp,
            fact_persona_coherence: fp_coh,
            repulsion_severity: branchless_max_f32(rep_sev, sys_sev),
            lineage_flags: prov_bits,
            attribution_tag: dominant_cause,
            dominant_slot: dom_slot,
            action_mode_flags: action_flags,
            active_swarm_cell_mask: active_swarm_mask,
            phase_step_min_q7: min_step_q7,
            subhead_fact_heads: subhead_fact,
            subhead_logic_heads: subhead_logic,
            subhead_explore_heads: subhead_explore,
            subhead_fact_gain_q7: fact_gain_q7,
            subhead_logic_gain_q7: logic_gain_q7,
            subhead_explore_gain_q7: pers_gain_q7,
            mean_phase_tag_q7: tag_q7,
            active_head_count: active_heads,
            frame_flags: packed_tag as u16,
            phase_deltas,
        },
        blended_w,
    )
}

/// Synthesizes a 128-byte [`GatewayFrame`] from optional **Origin 1 (`SystemConstraintSignal`)**
/// and **Origin 2 (`ObjectiveFactSignal`, `SubjectivePersonaSignal`, `ContradictionGuardSignal`)** components.
#[inline(always)]
#[must_use]
pub fn synthesize_multi_origin_gateway_frame<S, F, P, G>(
    sys_opt: Option<&S>,
    fact_opt: Option<&F>,
    persona_opt: Option<&P>,
    guard_opt: Option<&G>,
) -> GatewayFrame
where
    S: SystemConstraintSignal + ?Sized,
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
{
    synthesize_multi_origin_gateway_frame_with_words(sys_opt, fact_opt, persona_opt, guard_opt).0
}
