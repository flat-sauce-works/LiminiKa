// name: src/core/src/runtime/gateway/orchestrator/scale_layer.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-scale stigmergic orchestrator, hybrid circuits, and fractal scale-layer projections.

use super::*;
use crate::foundation::{
    RawPhaseProjectable, SWAR_SPLAT_BYTE_64, TaggedPtr, branchless_max_f32, swar_q7_abs_u8_x8,
    swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_scale_le128_with_bias_x8,
};
use crate::liminal::{
    LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES, PhaseQ7,
};
use crate::stigmergy::{
    CognitiveCircuitActivation, CognitiveOpKind, CognitiveStepOp, CognitiveTrajectoryLedger,
    LedgerSlot, PhaseAnchorRecord, PointerTrail, StigmergicReconstitutionReport,
    StigmergicScaleGranularity, StigmergicSubstrateOps, swar_mean_u8_le128_x8,
};

/// Blanket-implemented extension trait orchestrating [`StigmergicScaleLayer`] projection,
/// [`CognitiveTrajectoryLedger`], and scale-aware swarm resolution in zero-allocation mode.
pub trait ScaleLayerOrchestrator: StigmergicSubstrateOps + SubstrateGatewayExt {
    /// Projects `layer` across `input.input_words`, resolves the active pointer trail in the Micro-Swarm
    /// at the activated granularity tier, and records the emitted [`CognitiveStepOp`] into
    /// `ledger_slot.telemetry_mini_ledger`.
    ///
    /// # Errors
    /// Propagates any error from [`crate::stigmergy::PointerSubstrate::step_pointer`] or
    /// [`crate::stigmergy::FastPathLedger::telemetry_push`].
    #[inline]
    fn orchestrate_scale_layer(
        &mut self,
        layer: &(impl StigmergicScaleLayer + ?Sized),
        input: ScaleCascadeInput<'_>,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
        layer_index: usize,
        total_layers: usize,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        let mut ledger = CognitiveTrajectoryLedger::from_ledger_slot(ledger_slot);
        let proj = layer.project_scale_layer(input.input_words, &ledger, layer_index, total_layers);
        let eff_ptr = TaggedPtr(if input.input_ptr.is_null() {
            proj.frame.tagged_ptr
        } else {
            input.input_ptr.raw()
        });
        let (report, _resolved_w) = trail_out.resolve_swarm_with_scale_layer(
            eff_ptr,
            &proj.peeled_words,
            proj.granularity,
            layer_index,
            total_layers,
            proj.frame.active_swarm_cell_mask,
            input.bloom_filter,
        );
        self.step_pointer(report.evolved_ptr.raw(), trail_out)?;
        ledger.push_step(proj.emitted_step);
        ledger.write_to_ledger_slot(ledger_slot);
        ledger_slot.telemetry_ring_head = ledger_slot.telemetry_ring_head.wrapping_add(1);
        Ok((proj, ledger, report))
    }

    /// Executes an elastic `N`-layer cascade (`depth_n` in `1..=layers.len()`) across a slice of
    /// [`StigmergicScaleLayer`] stages using `AdaptiveThresholdPolicy::default()`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n<L: StigmergicScaleLayer>(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input: ScaleCascadeInput<'_>,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        let default_policy = AdaptiveThresholdPolicy::default();
        self.orchestrate_adaptive_depth_n_with_policy(
            layers,
            depth_n,
            input,
            &default_policy,
            ledger_slot,
            trail_out,
        )
    }

    /// Executes an elastic `N`-layer cascade (`depth_n` in `1..=layers.len()`) across a slice of
    /// [`StigmergicScaleLayer`] stages, deriving early-exit and recurrent-refinement thresholds
    /// from `circuit_policy`.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `layers` is empty or `depth_n == 0`, or
    /// propagates pointer/telemetry errors.
    #[inline]
    fn orchestrate_adaptive_depth_n_with_policy<
        L: StigmergicScaleLayer,
        P: CognitiveCircuitThresholdPolicy + ?Sized,
    >(
        &mut self,
        layers: &[L],
        depth_n: usize,
        input: ScaleCascadeInput<'_>,
        circuit_policy: &P,
        ledger_slot: &mut LedgerSlot,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(
        ScaleLayerProjection,
        CognitiveTrajectoryLedger,
        StigmergicReconstitutionReport,
    )> {
        #[inline(always)]
        const fn residual_mean_abs_q7(words: &[u64; NUM_STIGMERGIC_GRANULES]) -> u8 {
            let mut sum = 0u16;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                sum += swar_mean_u8_le128_x8(swar_q7_abs_u8_x8(words[c])) as u16;
                c += 1;
            }
            ((sum + 4) >> 3) as u8
        }

        let active_n = depth_n.min(layers.len());
        if active_n == 0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let fuse_floor = circuit_policy.early_exit_fuse_floor_q7();
        let repulse_ceil = circuit_policy.early_exit_repulse_ceiling_q7();
        let res_max = circuit_policy.early_exit_residual_max_q7();
        let refine_floor = circuit_policy.recurrent_refine_intensity_floor_q7();

        let mut last_proj =
            self.orchestrate_scale_layer(&layers[0], input, ledger_slot, trail_out, 0, active_n)?;
        let mut combined_peeled = last_proj.0.peeled_words;
        let mut residual = last_proj.0.residual_words;
        let mut ledger = last_proj.1;

        let mut i = 1usize;
        while i < active_n {
            let act = last_proj.0.activation;
            let converged_early = matches!(
                act.op_kind(),
                CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit
            ) && act.fuse_gate_q7 >= fuse_floor
                && act.repulse_sensitivity_q7 < repulse_ceil
                && residual_mean_abs_q7(&residual) <= res_max;
            if converged_early {
                break;
            }

            let step_res = self.orchestrate_scale_layer(
                &layers[i],
                ScaleCascadeInput {
                    input_ptr: last_proj.2.evolved_ptr,
                    input_words: &residual,
                    bloom_filter: input.bloom_filter,
                },
                ledger_slot,
                trail_out,
                i,
                active_n,
            )?;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                combined_peeled[c] =
                    swar_q7_circular_add_x8(combined_peeled[c], step_res.0.peeled_words[c]);
                c += 1;
            }
            residual = step_res.0.residual_words;
            ledger = step_res.1;
            last_proj = step_res;
            i += 1;
        }

        // Bounded single-step recurrent residual refinement when high-intensity peeling or guard
        // repulsion persists at the end of a multi-layer cascade.
        let tail_act = last_proj.0.activation;
        if active_n >= 2
            && matches!(
                tail_act.op_kind(),
                CognitiveOpKind::DecomposePeel | CognitiveOpKind::RepulseAudit
            )
            && tail_act.intensity_q7 >= refine_floor
            && residual_mean_abs_q7(&residual) > res_max
        {
            let refine_res = self.orchestrate_scale_layer(
                &layers[active_n - 1],
                ScaleCascadeInput {
                    input_ptr: last_proj.2.evolved_ptr,
                    input_words: &residual,
                    bloom_filter: input.bloom_filter,
                },
                ledger_slot,
                trail_out,
                active_n - 1,
                active_n,
            )?;
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                combined_peeled[c] =
                    swar_q7_circular_add_x8(combined_peeled[c], refine_res.0.peeled_words[c]);
                c += 1;
            }
            residual = refine_res.0.residual_words;
            ledger = refine_res.1;
            last_proj = refine_res;
        }

        let mut final_proj = last_proj.0;
        final_proj.peeled_words = combined_peeled;
        final_proj.residual_words = residual;
        final_proj.total_layers = active_n;
        Ok((final_proj, ledger, last_proj.2))
    }
}

impl<T: StigmergicSubstrateOps + SubstrateGatewayExt + ?Sized> ScaleLayerOrchestrator for T {}

/// Hierarchical scale-specialized hybrid circuit pairing a micro-scale (`PointerBit`/`CellTrail`)
/// [`CognitiveTrajectoryLedger`] with a macro-scale (`TypeGateway`/`MacroCoalition`) ledger.
///
/// Implements [`ManifoldGatewaySignal`] and [`StigmergicScaleLayer`] directly so callers can compose
/// analytical micro-peeling with integrative macro-fusion in zero-allocation cascades.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HybridScaleCircuit {
    /// Micro-scale circuit governing fine-grained decomposition and sub-head routing.
    pub micro_circuit: CognitiveTrajectoryLedger,
    /// Macro-scale circuit governing basin fusion and guard repulsion.
    pub macro_circuit: CognitiveTrajectoryLedger,
}

impl HybridScaleCircuit {
    /// Creates a new [`HybridScaleCircuit`] from distinct micro and macro trajectory ledgers.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        micro_circuit: CognitiveTrajectoryLedger,
        macro_circuit: CognitiveTrajectoryLedger,
    ) -> Self {
        Self {
            micro_circuit,
            macro_circuit,
        }
    }

    /// Selects the active ledger corresponding to `granularity`.
    #[inline(always)]
    #[must_use]
    pub const fn circuit_for_scale(
        &self,
        granularity: StigmergicScaleGranularity,
    ) -> &CognitiveTrajectoryLedger {
        match granularity {
            StigmergicScaleGranularity::PointerBit | StigmergicScaleGranularity::CellTrail => {
                &self.micro_circuit
            }
            StigmergicScaleGranularity::TypeGateway
            | StigmergicScaleGranularity::MacroCoalition => &self.macro_circuit,
        }
    }

    /// Flattens this hierarchical hybrid into a single blended 8-byte [`CognitiveTrajectoryLedger`].
    #[inline]
    #[must_use]
    pub fn to_blended_ledger(&self, macro_weight_q7: u8) -> CognitiveTrajectoryLedger {
        self.micro_circuit
            .compose_persona_circuit(&self.macro_circuit, macro_weight_q7)
    }
}

impl RawPhaseProjectable for HybridScaleCircuit {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.to_blended_ledger(64).project_to_q7_64()
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_blended_ledger(64).to_phase_words_64()
    }
}

impl CausalProvenanceSignal for HybridScaleCircuit {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.micro_circuit.causal_provenance_bits() | self.macro_circuit.causal_provenance_bits()
    }
}

impl CognitiveGroundingSignal for HybridScaleCircuit {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        branchless_max_f32(
            self.micro_circuit.grounding_base_energy(),
            self.macro_circuit.grounding_base_energy(),
        )
    }
}

impl SubjectivePersonaSignal for HybridScaleCircuit {}

impl ManifoldGatewaySignal for HybridScaleCircuit {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_blended_ledger(64).synthesize_gateway_frame()
    }
}

/// Output of projecting an input SWAR phase field through one or more [`StigmergicScaleLayer`] tiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleLayerProjection {
    /// Composite 128-byte [`GatewayFrame`] synthesized at this layer boundary.
    pub frame: GatewayFrame,
    /// Orthogonal contribution explained/peeled by this layer across 64 heads.
    pub peeled_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Remaining unexplained Q7 phase residual forwarded to layer `layer_index + 1`.
    pub residual_words: [u64; NUM_STIGMERGIC_GRANULES],
    /// Granularity tier executed at this projection step.
    pub granularity: StigmergicScaleGranularity,
    /// Zero-based starting layer index across the active stack (`0..total_layers`).
    pub layer_index: usize,
    /// Total effective layer count $N$ across the active cascade.
    pub total_layers: usize,
    /// Stigmergic cognitive gating activation computed from trajectory history and phase resonance.
    pub activation: CognitiveCircuitActivation,
    /// Compact 1-byte cognitive step emitted for recording into [`CognitiveTrajectoryLedger`].
    pub emitted_step: CognitiveStepOp,
}

/// Branchless core helper projecting `signal` at a specific [`StigmergicScaleGranularity`] and layer
/// depth ratio `layer_index / total_layers` with zero dynamic allocations.
#[inline]
pub fn project_gateway_signal_at_scale<S: ManifoldGatewaySignal + ?Sized>(
    signal: &S,
    granularity: StigmergicScaleGranularity,
    input_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ledger: &CognitiveTrajectoryLedger,
    layer_index: usize,
    total_layers: usize,
) -> ScaleLayerProjection {
    let safe_total = if total_layers == 0 { 1 } else { total_layers };
    let (mut frame, frm_w) = signal.synthesize_gateway_frame_with_words();
    let activation = ledger.evaluate_circuit_activation(&frm_w, input_words, granularity);
    let base_gain_q7 = granularity.layer_peel_gain_q7(layer_index, safe_total);
    let gate_mod = if activation.op_kind() == CognitiveOpKind::RepulseAudit {
        activation.repulse_sensitivity_q7
    } else {
        ((u16::from(base_gain_q7) + u16::from(activation.intensity_q7)) >> 1) as u8
    };
    let s_u64 = u64::from(gate_mod.min(128));
    let bias = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let active_mask = match granularity {
        StigmergicScaleGranularity::PointerBit => 1u8 << (layer_index & 7),
        StigmergicScaleGranularity::CellTrail => {
            if frame.active_swarm_cell_mask == 0 {
                0x0F
            } else {
                frame.active_swarm_cell_mask
            }
        }
        StigmergicScaleGranularity::TypeGateway | StigmergicScaleGranularity::MacroCoalition => {
            0xFF
        }
    };

    let mut peeled_words = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut residual_words = *input_words;
    let mut c = 0usize;
    while c < NUM_STIGMERGIC_GRANULES {
        if ((active_mask >> c) & 1) != 0 {
            let scaled = swar_q7_scale_le128_with_bias_x8(frm_w[c], s_u64, bias);
            peeled_words[c] = scaled;
            residual_words[c] = swar_q7_circular_diff_x8(input_words[c], scaled);
        }
        c += 1;
    }

    let eff_op = if frame.is_repulsed() {
        CognitiveOpKind::RepulseAudit
    } else {
        activation.op_kind()
    };
    let intensity_0_to_7 = (activation.intensity_q7 >> 4).min(7);
    let emitted_step = CognitiveStepOp::pack(granularity, eff_op, intensity_0_to_7);
    frame.active_swarm_cell_mask = active_mask;

    ScaleLayerProjection {
        frame,
        peeled_words,
        residual_words,
        granularity,
        layer_index,
        total_layers: safe_total,
        activation,
        emitted_step,
    }
}

/// Core Layer Management Trait rooted directly in [`ManifoldGatewaySignal`].
///
/// Governs multi-scale granularity (`PointerBit` -> `CellTrail` -> `TypeGateway` -> `MacroCoalition`),
/// structural depth composition (`N +/- 1` via tuples `(A, B)`, arrays `[T; N]`, or slices), and
/// orthogonal residual peeling guided by [`CognitiveTrajectoryLedger`].
pub trait StigmergicScaleLayer: ManifoldGatewaySignal {
    /// Returns the primary [`StigmergicScaleGranularity`] owned by this layer.
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }

    /// Returns the structural layer count `N >= 1` composed within this value.
    #[inline(always)]
    fn layer_count(&self) -> usize {
        1
    }

    /// Projects `input_words` through this layer at `(layer_index, total_layers)` using
    /// `ledger`, returning the peeled field, orthogonal residual, and emitted cognitive step.
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        project_gateway_signal_at_scale(
            self,
            self.scale_granularity(),
            input_words,
            ledger,
            layer_index,
            total_layers.max(self.layer_count()),
        )
    }
}

/// Zero-cost explicit-scale wrapper promoting any [`ManifoldGatewaySignal`] into a
/// [`StigmergicScaleLayer`] pinned to `granularity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleLayerNode<T> {
    /// Underlying signal implementing [`ManifoldGatewaySignal`].
    pub signal: T,
    /// Assigned [`StigmergicScaleGranularity`] tier inside a multi-layer stack.
    pub granularity: StigmergicScaleGranularity,
}

impl<T> ScaleLayerNode<T> {
    /// Wraps `signal` at `granularity`.
    #[inline(always)]
    #[must_use]
    pub const fn new(signal: T, granularity: StigmergicScaleGranularity) -> Self {
        Self {
            signal,
            granularity,
        }
    }
}

impl<T: CausalProvenanceSignal> CausalProvenanceSignal for ScaleLayerNode<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.signal.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        self.signal.causal_anomaly_severity()
    }
}

impl<T: ManifoldGatewaySignal> ManifoldGatewaySignal for ScaleLayerNode<T> {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.signal.synthesize_gateway_frame()
    }

    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.signal.synthesize_gateway_frame_with_words()
    }
}

impl<T: ManifoldGatewaySignal> StigmergicScaleLayer for ScaleLayerNode<T> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.granularity
    }
}

impl<T: StigmergicScaleLayer + ?Sized> StigmergicScaleLayer for &T {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        (**self).scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        (**self).layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        (**self).project_scale_layer(input_words, ledger, layer_index, total_layers)
    }
}

impl<
    A: StigmergicScaleLayer + ObjectiveFactSignal,
    B: StigmergicScaleLayer + SubjectivePersonaSignal,
> StigmergicScaleLayer for (A, B)
{
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        let ga = self.0.scale_granularity();
        let gb = self.1.scale_granularity();
        if gb > ga { gb } else { ga }
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        self.0.layer_count() + self.1.layer_count()
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        let a_layers = self.0.layer_count();
        let total = total_layers.max(self.layer_count());
        let proj_a = self
            .0
            .project_scale_layer(input_words, ledger, layer_index, total);
        let mut mid_ledger = *ledger;
        mid_ledger.push_step(proj_a.emitted_step);
        let proj_b = self.1.project_scale_layer(
            &proj_a.residual_words,
            &mid_ledger,
            layer_index + a_layers,
            total,
        );
        let (frame, _w) = self.synthesize_gateway_frame_with_words();
        let mut combined_peeled = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            combined_peeled[c] =
                swar_q7_circular_add_x8(proj_a.peeled_words[c], proj_b.peeled_words[c]);
            c += 1;
        }
        ScaleLayerProjection {
            frame,
            peeled_words: combined_peeled,
            residual_words: proj_b.residual_words,
            granularity: proj_b.granularity,
            layer_index,
            total_layers: total,
            activation: proj_b.activation,
            emitted_step: proj_b.emitted_step,
        }
    }
}

impl<T: StigmergicScaleLayer> StigmergicScaleLayer for [T] {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        if self.is_empty() {
            StigmergicScaleGranularity::TypeGateway
        } else {
            self[0].scale_granularity()
        }
    }

    #[inline]
    fn layer_count(&self) -> usize {
        let mut total = 0usize;
        let mut i = 0usize;
        while i < self.len() {
            total += self[i].layer_count();
            i += 1;
        }
        total.max(1)
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        if self.is_empty() {
            return ScaleLayerProjection {
                frame: GatewayFrame::default(),
                peeled_words: [0u64; NUM_STIGMERGIC_GRANULES],
                residual_words: *input_words,
                granularity: StigmergicScaleGranularity::TypeGateway,
                layer_index,
                total_layers: total_layers.max(1),
                activation: CognitiveCircuitActivation::default(),
                emitted_step: CognitiveStepOp::default(),
            };
        }
        let own_layers = self.layer_count();
        let total = total_layers.max(own_layers);
        let mut cur_residual = *input_words;
        let mut cur_ledger = *ledger;
        let mut combined_peeled = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut cursor = layer_index;
        let mut last_proj = self[0].project_scale_layer(&cur_residual, &cur_ledger, cursor, total);
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            combined_peeled[c] = last_proj.peeled_words[c];
            c += 1;
        }
        cur_residual = last_proj.residual_words;
        cur_ledger.push_step(last_proj.emitted_step);
        cursor += self[0].layer_count();
        let mut i = 1usize;
        while i < self.len() {
            let p = self[i].project_scale_layer(&cur_residual, &cur_ledger, cursor, total);
            let mut k = 0usize;
            while k < NUM_STIGMERGIC_GRANULES {
                combined_peeled[k] = swar_q7_circular_add_x8(combined_peeled[k], p.peeled_words[k]);
                k += 1;
            }
            cur_residual = p.residual_words;
            cur_ledger.push_step(p.emitted_step);
            cursor += self[i].layer_count();
            last_proj = p;
            i += 1;
        }
        let (frame, _w) = self.synthesize_gateway_frame_with_words();
        ScaleLayerProjection {
            frame,
            peeled_words: combined_peeled,
            residual_words: cur_residual,
            granularity: last_proj.granularity,
            layer_index,
            total_layers: total,
            activation: last_proj.activation,
            emitted_step: last_proj.emitted_step,
        }
    }
}

impl<T: StigmergicScaleLayer, const N: usize> StigmergicScaleLayer for [T; N] {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.as_slice().scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        self.as_slice().layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        self.as_slice()
            .project_scale_layer(input_words, ledger, layer_index, total_layers)
    }
}

impl StigmergicScaleLayer for GatewayFrame {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for ObjectiveFact<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::CellTrail
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for SubjectivePersona<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for GuardPhaseAnchor {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::TypeGateway
    }
}

impl StigmergicScaleLayer for PhaseAnchorRecord {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        self.latest_step()
            .map(CognitiveStepOp::granularity)
            .unwrap_or(StigmergicScaleGranularity::TypeGateway)
    }
}

impl StigmergicScaleLayer for HybridScaleCircuit {
    #[inline(always)]
    fn scale_granularity(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::CellTrail
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        2
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        let g = StigmergicScaleGranularity::from_layer_ratio(layer_index, total_layers);
        let active_circuit = self.circuit_for_scale(g);
        let merged = active_circuit.compose_persona_circuit(ledger, 64);
        project_gateway_signal_at_scale(
            active_circuit,
            g,
            input_words,
            &merged,
            layer_index,
            total_layers,
        )
    }
}
