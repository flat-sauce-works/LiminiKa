// name: src/core/src/runtime/gateway/frame/adapters.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Blanket [`ManifoldGatewaySignal`] adapter implementations and [`ResonantSignalSlice`].

use super::super::*;
use super::signal::{
    coherence_for_active_heads_64, fold_gateway_slice_with_query_words_and_heads_core,
    tangent_cosine_active_heads_64,
};
use super::synthesis::{
    apply_sys_and_guard_to_pair_frame, combine_fact_persona_aperture_frame_pure,
    synthesize_multi_origin_gateway_frame_with_words,
};
use super::{GatewayFrame, ManifoldGatewaySignal};
use crate::drivers::EntropyState;
use crate::foundation::{
    RawPhaseProjectable, branchless_clamp_f32, branchless_max_f32, is_word_array_uniform_64,
    load_q7_words_64, swar_words_nonzero_64,
};
use crate::liminal::{
    MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES, PhaseQ7, RuntimeConfig, branchless_select_f32,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::context::CoreStateStack;
use crate::runtime::horizon::EpistemicHorizonReading;
use crate::stigmergy::{OverlapDiscoveryReport, PointerTrail, trail_phase_words_ref};

// ===================================================================
// 7. Blanket `ManifoldGatewaySignal` Implementations for Wrappers, Tuples & Ensembles
// ===================================================================

impl<T> ManifoldGatewaySignal for SystemConstraint<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl ManifoldGatewaySignal for EntropyState {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl ManifoldGatewaySignal for RuntimeConfig {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            Self,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(Some(self), None, None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for ObjectiveFact<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            Self,
            SubjectivePersona<[u8; 0]>,
            ContradictionGuard<()>,
        >(None, Some(self), None, None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<T: RawPhaseProjectable> ManifoldGatewaySignal for SubjectivePersona<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>,
            Self,
            ContradictionGuard<()>,
        >(None, None, Some(self), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<B> ManifoldGatewaySignal for ContradictionGuard<B>
where
    Self: ContradictionGuardSignal,
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            Self,
        >(None, None, None, Some(self))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl ManifoldGatewaySignal for GuardPhaseAnchor {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            ObjectiveFact<[u8; 0]>,
            SubjectivePersona<[u8; 0]>,
            Self,
        >(None, None, None, Some(self))
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<F: ObjectiveFactSignal, P: SubjectivePersonaSignal> CausalProvenanceSignal for (F, P) {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits()
            | self.1.causal_provenance_bits()
            | CAUSAL_AXIS_OBJECTIVE_FACT
            | CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            self.0.causal_anomaly_severity(),
            self.1.causal_anomaly_severity(),
        )
    }
}

impl<F: ObjectiveFactSignal, P: SubjectivePersonaSignal> ManifoldGatewaySignal for (F, P) {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<
            SystemConstraint<()>,
            F,
            P,
            ContradictionGuard<()>,
        >(None, Some(&self.0), Some(&self.1), None)
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut combined = combine_fact_persona_aperture_frame_pure(&self.0, &self.1, stack);
        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut combined,
        );
        let words = *combined.phase_words_ref();
        (combined, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let mut combined = combine_fact_persona_aperture_frame_pure(&self.0, &self.1, stack);
        stack.modulate_gateway_frame_with_ego_readonly(
            delta,
            horizon,
            overlap_report,
            &mut combined,
        );
        let words = *combined.phase_words_ref();
        (combined, words)
    }
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    CausalProvenanceSignal for (S, F, P)
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits()
            | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits()
            | CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | CAUSAL_AXIS_OBJECTIVE_FACT
            | CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            self.0.causal_anomaly_severity(),
            branchless_max_f32(
                self.1.causal_anomaly_severity(),
                self.2.causal_anomaly_severity(),
            ),
        )
    }
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    ManifoldGatewaySignal for (S, F, P)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, ContradictionGuard<()>>(
            Some(&self.0),
            Some(&self.1),
            Some(&self.2),
            None,
        )
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let mut guarded = apply_sys_and_guard_to_pair_frame::<S, ContradictionGuard<()>>(
            &self.0, None, pair_frame,
        );
        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut guarded,
        );
        let words = *guarded.phase_words_ref();
        (guarded, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let mut guarded = apply_sys_and_guard_to_pair_frame::<S, ContradictionGuard<()>>(
            &self.0, None, pair_frame,
        );
        stack.modulate_gateway_frame_with_ego_readonly(
            delta,
            horizon,
            overlap_report,
            &mut guarded,
        );
        let words = *guarded.phase_words_ref();
        (guarded, words)
    }
}

impl<
    S: SystemConstraintSignal,
    F: ObjectiveFactSignal,
    P: SubjectivePersonaSignal,
    G: ContradictionGuardSignal,
> CausalProvenanceSignal for (S, F, P, G)
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits()
            | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits()
            | self.3.causal_provenance_bits()
            | CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | CAUSAL_AXIS_OBJECTIVE_FACT
            | CAUSAL_AXIS_SUBJECTIVE_PERSONA
            | CAUSAL_AXIS_CONTRADICTION_GUARD
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(
            branchless_max_f32(
                self.0.causal_anomaly_severity(),
                self.1.causal_anomaly_severity(),
            ),
            branchless_max_f32(
                self.2.causal_anomaly_severity(),
                self.3.causal_anomaly_severity(),
            ),
        )
    }
}

impl<
    S: SystemConstraintSignal,
    F: ObjectiveFactSignal,
    P: SubjectivePersonaSignal,
    G: ContradictionGuardSignal,
> ManifoldGatewaySignal for (S, F, P, G)
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        synthesize_multi_origin_gateway_frame_with_words::<S, F, P, G>(
            Some(&self.0),
            Some(&self.1),
            Some(&self.2),
            Some(&self.3),
        )
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let mut guarded = apply_sys_and_guard_to_pair_frame(&self.0, Some(&self.3), pair_frame);
        stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            overlap_report,
            &mut guarded,
        );
        let words = *guarded.phase_words_ref();
        (guarded, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &CoreStateStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let mut guarded = apply_sys_and_guard_to_pair_frame(&self.0, Some(&self.3), pair_frame);
        stack.modulate_gateway_frame_with_ego_readonly(
            delta,
            horizon,
            overlap_report,
            &mut guarded,
        );
        let words = *guarded.phase_words_ref();
        (guarded, words)
    }
}

impl<T: ManifoldGatewaySignal> CausalProvenanceSignal for Option<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        match self {
            Some(inner) => inner.causal_provenance_bits(),
            None => 0,
        }
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        match self {
            Some(inner) => inner.causal_anomaly_severity(),
            None => 0.0,
        }
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for Option<T> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        match self {
            Some(inner) => inner.synthesize_gateway_frame_with_words(),
            None => (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]),
        }
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        match self {
            Some(inner) => inner.has_active_gateway_ensemble_signal(),
            None => false,
        }
    }
}

impl<T: ManifoldGatewaySignal> CausalProvenanceSignal for [T] {
    #[inline]
    fn causal_provenance_bits(&self) -> u32 {
        let mut bits = 0u32;
        let mut i = 0usize;
        while i < self.len() {
            bits |= self[i].causal_provenance_bits();
            i += 1;
        }
        bits
    }
    #[inline]
    fn causal_anomaly_severity(&self) -> f32 {
        let mut max_sev = 0.0_f32;
        let mut i = 0usize;
        while i < self.len() {
            max_sev = branchless_max_f32(max_sev, self[i].causal_anomaly_severity());
            i += 1;
        }
        max_sev
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for [T] {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        T::synthesize_gateway_slice_with_words(self)
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        T::synthesize_gateway_slice_with_words(self).0
    }
}

impl<T: ManifoldGatewaySignal, const N: usize> CausalProvenanceSignal for [T; N] {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.as_slice().causal_provenance_bits()
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.as_slice().causal_anomaly_severity()
    }
}

impl<T: ManifoldGatewaySignal, const N: usize> ManifoldGatewaySignal for [T; N] {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.as_slice().synthesize_gateway_frame_with_words()
    }
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.as_slice().synthesize_gateway_frame_with_words().0
    }
}

/// Arbitrary-length [`ManifoldGatewaySignal`] slice adapter that weights each element by its
/// $SO(2)^{H}$ phase resonance against `query_phase_words` over `active_heads` active heads
/// (zero heap allocation, zero fixed count).
pub struct ResonantSignalSlice<'a, S: ManifoldGatewaySignal> {
    /// Dynamic slice of candidate Gateway signals (e.g. facts, personas, or attractor records).
    pub signals: &'a [S],
    /// Query-side 64-head Q7 phase words packed into 8 SWAR `u64` lanes.
    pub query_phase_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Active attention-head count $H \in [1, 64]$ of the query probe (excludes zero-padded lanes).
    pub active_heads: u16,
}

impl<'a, S: ManifoldGatewaySignal> ResonantSignalSlice<'a, S> {
    /// Wraps an arbitrary slice `signals` with a Q7 phase probe slice (`<= 64` heads).
    #[inline]
    #[must_use]
    pub fn from_q7_probe(signals: &'a [S], probe_q7: &[PhaseQ7]) -> Self {
        let mut arr = [0i8; MAX_ATTENTION_HEADS];
        let limit = probe_q7.len().min(MAX_ATTENTION_HEADS);
        arr[..limit].copy_from_slice(&probe_q7[..limit]);
        Self {
            signals,
            query_phase_words: load_q7_words_64(&arr),
            active_heads: limit.clamp(1, MAX_ATTENTION_HEADS) as u16,
        }
    }

    /// Wraps an arbitrary slice `signals` using the accumulated phase words of `trail`.
    #[inline]
    #[must_use]
    pub fn from_trail(signals: &'a [S], trail: &PointerTrail) -> Self {
        Self {
            signals,
            query_phase_words: *trail_phase_words_ref(trail),
            active_heads: MAX_ATTENTION_HEADS as u16,
        }
    }
}

impl<S: ManifoldGatewaySignal> CausalProvenanceSignal for ResonantSignalSlice<'_, S> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.signals.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.signals.causal_anomaly_severity()
    }
}

impl<S: ManifoldGatewaySignal> ManifoldGatewaySignal for ResonantSignalSlice<'_, S> {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        fold_gateway_slice_with_query_words_and_heads_core(
            self.signals,
            Some(&self.query_phase_words),
            self.active_heads,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<S: CognitiveGroundingSignal + ManifoldGatewaySignal> CognitiveGroundingSignal
    for ResonantSignalSlice<'_, S>
{
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.synthesize_gateway_frame().phase_deltas
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.synthesize_gateway_frame_with_words().1
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.synthesize_gateway_frame().effective_energy
    }

    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.synthesize_gateway_frame().mapped_section_offset48
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.synthesize_gateway_frame().target_anchor_id
    }

    #[inline(always)]
    fn active_head_count(&self) -> u16 {
        self.active_heads
    }

    #[inline(always)]
    fn synthesize_grounding_envelope(
        &self,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
        let (frame, words) = self.synthesize_gateway_frame_with_words();
        (
            words,
            frame.effective_energy,
            frame.mapped_section_offset48,
            frame.target_anchor_id,
            frame.active_head_count,
        )
    }
}

impl<S: ObjectiveFactSignal + ManifoldGatewaySignal> ObjectiveFactSignal
    for ResonantSignalSlice<'_, S>
{
}

impl<S: SubjectivePersonaSignal + ManifoldGatewaySignal> SubjectivePersonaSignal
    for ResonantSignalSlice<'_, S>
{
}

impl<S: ContradictionGuardSignal + CognitiveGroundingSignal + ManifoldGatewaySignal>
    ContradictionGuardSignal for ResonantSignalSlice<'_, S>
{
    #[inline]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let candidate_words = candidate_phases.project_to_words_64();
        self.evaluate_repulsion_severity_from_words(&candidate_words, candidate_phases)
    }

    #[inline]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let mut max_coh = -1.0_f32;
        let mut max_sev = 0.0_f32;
        let is_subspace = self.active_heads < MAX_ATTENTION_HEADS as u16;
        let query_uni = is_word_array_uniform_64(&self.query_phase_words);
        for sig in self.signals {
            let sig_sev =
                sig.evaluate_repulsion_severity_from_words(candidate_words, candidate_phases);
            max_sev = branchless_max_f32(max_sev, sig_sev);
            let sig_words = sig.project_grounding_words_64();
            let coh = if is_subspace {
                tangent_cosine_active_heads_64(
                    &self.query_phase_words,
                    &sig_words,
                    self.active_heads,
                )
            } else {
                let sig_uni = is_word_array_uniform_64(&sig_words);
                coherence_for_active_heads_64(
                    &self.query_phase_words,
                    query_uni,
                    &sig_words,
                    sig_uni,
                    self.active_heads,
                )
            };
            max_coh = branchless_max_f32(max_coh, coh);
        }
        // Trigger repulsion only when the query probe resonates very strongly with a guard anchor.
        let res_sev = branchless_select_f32(
            max_coh > 0.90_f32,
            branchless_clamp_f32(0.55_f32 + (max_coh - 0.90_f32) * 4.5_f32, 0.55, 1.0),
            0.0,
        );
        branchless_max_f32(max_sev, res_sev)
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }

    #[inline]
    fn project_repulsive_pole_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let mut best_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut best_score = -2.0_f32;
        let is_subspace = self.active_heads < MAX_ATTENTION_HEADS as u16;
        let query_uni = is_word_array_uniform_64(&self.query_phase_words);
        for sig in self.signals {
            let sig_words = sig.project_repulsive_pole_words_64();
            if !swar_words_nonzero_64(&sig_words) {
                continue;
            }
            let coh = if is_subspace {
                tangent_cosine_active_heads_64(
                    &self.query_phase_words,
                    &sig_words,
                    self.active_heads,
                )
            } else {
                let sig_uni = is_word_array_uniform_64(&sig_words);
                coherence_for_active_heads_64(
                    &self.query_phase_words,
                    query_uni,
                    &sig_words,
                    sig_uni,
                    self.active_heads,
                )
            };
            let score = coh + branchless_select_f32(sig.causal_anomaly_severity() > 0.50, 2.0, 0.0);
            if score > best_score {
                best_score = score;
                best_words = sig_words;
            }
        }
        best_words
    }
}
