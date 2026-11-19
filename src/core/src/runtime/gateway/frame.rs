// name: src/core/src/runtime/gateway/frame.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single Composite Gateway Frame ([`GatewayFrame`] 128B `Pod`), [`ManifoldGatewaySignal`],
//! multi-origin synthesis, and ensemble folding.

use super::*;

// ===================================================================
// 6. Single Composite Gateway Frame (`GatewayFrame` 128B Pod) & Trait
// ===================================================================

/// 128-byte (2-cacheline, `#[repr(C, align(64))]`) [`Pod`] composite descriptor produced by
/// [`ManifoldGatewaySignal::synthesize_gateway_frame`].
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GatewayFrame {
    /// Synthesized 64-bit tagged pointer (`[Phase(8b) | Slot(3b) | ActionMode(5b) | ZimmsAddr(48b)]`, `offset 0..8`).
    pub tagged_ptr: u64,
    /// Direct 48-bit `.gcso` (mapped-storage) section offset (`bits 0..=47`, `offset 8..16`).
    pub mapped_section_offset48: u64,
    /// Full 32-bit macro attractor anchor ID on the Single Unified Attractor Manifold (`offset 16..20`).
    pub target_anchor_id: u32,
    /// Combined gravitational pull energy $E_{\mathrm{eff}} \ge 0$ (`offset 20..24`).
    pub effective_energy: f32,
    /// Normalized RIPA soft-clamping limit in radians ($\theta_{\max}$, `offset 24..28`).
    pub ripa_clamp_rad: f32,
    /// QDPS minimum step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$, `offset 28..32`).
    pub qdps_min_step_rad: f32,
    /// Multiplicative inverse-temperature logit scale $\beta_{\mathrm{eff}} \in [0.25, 4.0]$ (`offset 32..36`).
    pub effective_inv_temp: f32,
    /// Cosine phase coherence $\mathcal{C}_{\mathrm{FP}} \in [-1, 1]$ between Objective Fact and Subjective Persona (`offset 36..40`).
    pub fact_persona_coherence: f32,
    /// Repulsive anomaly severity in $[0.0, 1.0]$ (`> 0.5` indicates anti-phase repulsion, `offset 40..44`).
    pub repulsion_severity: f32,
    /// Bitwise OR of active origin axes (`CAUSAL_AXIS_*`) and diagnostic causes (`CAUSAL_CAUSE_*`, `offset 44..48`).
    pub provenance_bits: u32,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`, `offset 48`).
    pub dominant_cause_code: u8,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`, `offset 49`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags (`offset 50`).
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`, `offset 51`).
    pub active_swarm_cell_mask: u8,
    /// Ceiling-rounded Q7 integer step filter cutoff (`0..=128`, `offset 52`).
    pub qdps_min_step_q7: u8,
    /// Recommended sub-head router `Fact` sub-head count (`offset 53`).
    pub subhead_fact_heads: u8,
    /// Recommended sub-head router `Logic` sub-head count (`offset 54`).
    pub subhead_logic_heads: u8,
    /// Recommended sub-head router `Explore` sub-head count (`offset 55`).
    pub subhead_explore_heads: u8,
    /// Quantized sub-head router `Fact` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 56`).
    pub subhead_fact_gain_q7: u8,
    /// Quantized sub-head router `Logic` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 57`).
    pub subhead_logic_gain_q7: u8,
    /// Quantized sub-head router `Explore` sub-head phase gain in Q7 (`128` = `1.0x`, `offset 58`).
    pub subhead_explore_gain_q7: u8,
    /// Fast 64-head SWAR arithmetic mean Q7 phase tag $\bar{\theta}_{Q7} \in [-127, 127]$ (`offset 59`).
    pub mean_phase_tag_q7: gcso_q7_t,
    /// Active attention head count for this frame (`1..=64`, defaults to `64`, `offset 60..62`).
    pub active_head_count: u16,
    /// Frame status and routing telemetry flags (`offset 62..64`).
    pub frame_flags: u16,
    /// Synthesized 64-head Q7 phase delta profile on $SO(2)^{64}$ (`64` bytes at `offset 64..128`).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

// SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` of exact size 128 bytes with zero uninitialized padding.
unsafe impl Pod for GatewayFrame {}

impl Default for GatewayFrame {
    #[inline(always)]
    fn default() -> Self {
        Self {
            tagged_ptr: gcso_tagged_ptr_t::pack(
                0,
                pack_action_slot_tag(ATTRACTOR_SLOT_SYSTEM_BASELINE, ACTION_PRESET_EXPRESS_PHASE),
                0,
            )
            .raw(),
            mapped_section_offset48: 0,
            target_anchor_id: 0,
            effective_energy: 0.0,
            ripa_clamp_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_PHASE_STEP_MIN_RAD,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0,
            repulsion_severity: 0.0,
            provenance_bits: 0,
            dominant_cause_code: ROOT_CAUSE_NOMINAL,
            dominant_slot: ATTRACTOR_SLOT_SYSTEM_BASELINE,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: 0xFF,
            qdps_min_step_q7: branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD),
            subhead_fact_heads: 24,
            subhead_logic_heads: 24,
            subhead_explore_heads: 16,
            subhead_fact_gain_q7: 128,
            subhead_logic_gain_q7: 128,
            subhead_explore_gain_q7: 128,
            mean_phase_tag_q7: 0,
            active_head_count: MAX_ATTENTION_HEADS as u16,
            frame_flags: 0,
            phase_deltas: [0i8; MAX_ATTENTION_HEADS],
        }
    }
}

impl GatewayFrame {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 64..128`,
    /// guaranteed 64-byte aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; NUM_STIGMERGIC_GRANULES] {
        // SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe { &*(self.phase_deltas.as_ptr() as *const [u64; NUM_STIGMERGIC_GRANULES]) }
    }

    /// Borrows `self.phase_deltas` (`offset 64` of `#[repr(C, align(64))]` [`GatewayFrame`],
    /// guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR word array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; NUM_STIGMERGIC_GRANULES] {
        // SAFETY: `GatewayFrame` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe { &mut *(self.phase_deltas.as_mut_ptr() as *mut [u64; NUM_STIGMERGIC_GRANULES]) }
    }

    /// Returns `true` when this [`GatewayFrame`] carries an active cohomology or Bloom repulsion guard
    /// that must be rejected before mutating a [`CrystallizedAttractorRecord`] centroid.
    #[inline(always)]
    #[must_use]
    pub const fn is_repulsed(&self) -> bool {
        (self.repulsion_severity > 0.50)
            | ((self.action_mode_flags & ACTION_MODE_PHASE_REPULSE) != 0)
            | (self.dominant_cause_code == ROOT_CAUSE_COHOMOLOGY_REPULSION)
            | ((self.provenance_bits & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0)
    }

    /// Resolves the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, even
    /// integer in `2..=head_dim`) directly from this [`GatewayFrame`]'s `provenance_bits` and
    /// `active_swarm_cell_mask` combined with `moving_z_entropy` and `von_mises_kappa`.
    #[inline(always)]
    #[must_use]
    pub const fn resolve_autonomous_rot_dim(
        &self,
        head_dim: usize,
        moving_z_entropy: f32,
        von_mises_kappa: f32,
    ) -> usize {
        resolve_autonomous_swarm_rot_dim(
            head_dim,
            self.provenance_bits,
            self.active_swarm_cell_mask,
            moving_z_entropy,
            von_mises_kappa,
        )
    }

    /// Extracts the 16-byte [`CausalProvenanceReport`] from this [`GatewayFrame`] in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn to_causal_report(&self) -> CausalProvenanceReport {
        CausalProvenanceReport {
            provenance_bits: self.provenance_bits,
            fact_persona_coherence: self.fact_persona_coherence,
            anomaly_severity: self.repulsion_severity,
            dominant_cause_code: self.dominant_cause_code,
            dominant_slot: self.dominant_slot,
            action_mode_flags: self.action_mode_flags,
            active_swarm_mask: self.active_swarm_cell_mask,
        }
    }

    /// Converts the Gateway's recommended Fact/Logic/Explore head split into a [`gcso_subhead_config_t`]
    /// scaled proportionally to `total_heads`.
    #[inline(always)]
    #[must_use]
    pub const fn to_subhead_config_for_heads(&self, total_heads: usize) -> gcso_subhead_config_t {
        let h = if total_heads > MAX_ATTENTION_HEADS {
            MAX_ATTENTION_HEADS
        } else {
            total_heads
        };
        if h == 0 {
            return gcso_subhead_config_t {
                num_fact_heads: 0,
                num_logic_heads: 0,
                num_explore_heads: 0,
                flags: 0,
                fact_phase_gain: 1.0,
                logic_phase_gain: 1.0,
                explore_phase_gain: 1.0,
                reserved: [0; 12],
            };
        }
        let f_raw = ((self.subhead_fact_heads as usize) * h + 32) >> 6;
        let l_raw = ((self.subhead_logic_heads as usize) * h + 32) >> 6;
        let f_cnt = if f_raw > h { h } else { f_raw };
        let rem = h - f_cnt;
        let l_cnt = if l_raw > rem { rem } else { l_raw };
        let e_cnt = h - f_cnt - l_cnt;
        gcso_subhead_config_t {
            num_fact_heads: f_cnt as u16,
            num_logic_heads: l_cnt as u16,
            num_explore_heads: e_cnt as u16,
            flags: 1,
            fact_phase_gain: (self.subhead_fact_gain_q7 as f32) * (1.0 / 128.0),
            logic_phase_gain: (self.subhead_logic_gain_q7 as f32) * (1.0 / 128.0),
            explore_phase_gain: (self.subhead_explore_gain_q7 as f32) * (1.0 / 128.0),
            reserved: [0; 12],
        }
    }

    /// Returns `true` when all three sub-head router sub-head group gains (`subhead_fact_gain_q7`,
    /// `subhead_logic_gain_q7`, `subhead_explore_gain_q7`) equal unit gain (`128` in Q0.7).
    #[inline(always)]
    #[must_use]
    pub const fn has_unit_subhead_gains(&self) -> bool {
        (self.subhead_fact_gain_q7 == 128)
            & (self.subhead_logic_gain_q7 == 128)
            & (self.subhead_explore_gain_q7 == 128)
    }

    /// Autonomously derives the 8-cell Micro-Swarm step gain (`32..=128` in Q0.7) from this
    /// [`GatewayFrame`]'s `effective_energy` and `fact_persona_coherence` without manual tuning.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_step_gain_u7(&self) -> u8 {
        let e = branchless_clamp_f32(branchless_relu_f32(self.effective_energy), 0.0, 4.0);
        let coh = branchless_clamp_f32(self.fact_persona_coherence, 0.0, 1.0);
        let scaled = 48.0 + e * 16.0 + coh * 16.0;
        let clamped = branchless_clamp_f32(scaled, 32.0, 128.0);
        (clamped + 0.5) as u8
    }

    /// Autonomously derives the Micro-Swarm phase convergence tolerance in Q0.7 (`8..=20`) from
    /// this [`GatewayFrame`]'s active causal origins (`provenance_bits`).
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_tolerance_q7(&self) -> u8 {
        let has_fact = (self.provenance_bits & CAUSAL_AXIS_OBJECTIVE_FACT) != 0;
        let has_persona = (self.provenance_bits & CAUSAL_AXIS_SUBJECTIVE_PERSONA) != 0;
        let base = branchless_select_u8(has_persona, 16, 12);
        branchless_select_u8(has_fact, 10, base)
    }

    /// Autonomously derives the Micro-Swarm quorum cell count threshold (`4..=7` out of 8 cells)
    /// from this [`GatewayFrame`]'s active causal origins and repulsion severity.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_swarm_quorum_threshold(&self) -> u8 {
        let strict = ((self.provenance_bits
            & (CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_CONTRADICTION_GUARD))
            != 0)
            | (self.repulsion_severity > 0.25);
        branchless_select_u8(strict, 6, 5)
    }

    /// Autonomously derives the unresolved-cell damping factor (`32..=96` in Q0.7) from this
    /// [`GatewayFrame`]'s `repulsion_severity`.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_unresolved_damping_u7(&self) -> u8 {
        branchless_select_u8(self.is_repulsed(), 32, 64)
    }

    /// Autonomously derives the Bayesian Limina geodesic interpolation weight (`32..=96` in Q0.7)
    /// from this [`GatewayFrame`]'s `fact_persona_coherence`.
    #[inline(always)]
    #[must_use]
    pub const fn autonomous_liminal_interpolation_u7(&self) -> u8 {
        let coh = branchless_clamp_f32(self.fact_persona_coherence, -1.0, 1.0);
        let scaled = 64.0 + coh * 32.0;
        (branchless_clamp_f32(scaled, 32.0, 96.0) + 0.5) as u8
    }

    /// Rotates a 32-float `(Q, K, V)` block triple in-place by the RIPA-clamped $SO(2)$ phase angle
    /// of `head_idx` (`0..64`).
    #[inline(always)]
    pub fn rotate_qkv_block32_in_place(
        &self,
        q_block32: &mut [f32; 32],
        k_block32: &mut [f32; 32],
        v_block32: &mut [f32; 32],
        head_idx: usize,
    ) {
        let phase_q7 = self.phase_deltas[head_idx & (MAX_ATTENTION_HEADS - 1)];
        let (sin_t, cos_t) = branchless_sincos_q7_phase_clamp(phase_q7, self.ripa_clamp_rad);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `target_feature = "avx2"` is enabled and `q_block32`, `k_block32`, `v_block32` are disjoint 32-float arrays.
            unsafe {
                use core::arch::x86_64::_mm256_set1_ps;
                crate::foundation::branchless::avx2_rotate_so2_triple_block32(
                    q_block32.as_mut_ptr(),
                    k_block32.as_mut_ptr(),
                    v_block32.as_mut_ptr(),
                    _mm256_set1_ps(cos_t),
                    _mm256_set1_ps(sin_t),
                );
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        {
            let mut idx = 0usize;
            while idx < 32 {
                let (q0, q1) =
                    branchless_rotate_so2_pair(q_block32[idx], q_block32[idx + 1], sin_t, cos_t);
                q_block32[idx] = q0;
                q_block32[idx + 1] = q1;
                let (k0, k1) =
                    branchless_rotate_so2_pair(k_block32[idx], k_block32[idx + 1], sin_t, cos_t);
                k_block32[idx] = k0;
                k_block32[idx + 1] = k1;
                let (v0, v1) =
                    branchless_rotate_so2_pair(v_block32[idx], v_block32[idx + 1], sin_t, cos_t);
                v_block32[idx] = v0;
                v_block32[idx + 1] = v1;
                idx += 2;
            }
        }
    }
}

/// **The Single Composite Manifold Gateway Trait (`ManifoldGatewaySignal: CausalProvenanceSignal`)**.
pub trait ManifoldGatewaySignal: CausalProvenanceSignal {
    /// Wide composite cognitive aperture combining causal origins, circuit plasticity, KV phase
    /// cache, relative coupling, and dissipative crystallization (`0x0000_00FF`).
    const GATEWAY_APERTURE_MASK: u32 = THRESHOLD_MANIFOLD_GATEWAY;

    /// Complement bitmask outside the Gateway horizon (`!THRESHOLD_MANIFOLD_GATEWAY`).
    const GATEWAY_UNKNOWABLE_VOID_MASK: u32 = !Self::GATEWAY_APERTURE_MASK;

    /// Branchlessly partitions `self.causal_provenance_bits()` into
    /// `(cognized_bits, unknowable_void_bits)` for [`Self::GATEWAY_APERTURE_MASK`].
    #[inline(always)]
    fn partition_gateway_horizon(&self) -> (u32, u32) {
        self.partition_cognitive_and_unknowable_bits(
            Self::GATEWAY_APERTURE_MASK,
            self.causal_provenance_bits(),
        )
    }

    /// Synthesizes all constituent blanket signals into a 128-byte [`GatewayFrame`] in
    /// constant-time branchless arithmetic ($C_{\mathrm{alloc}} = 0$).
    fn synthesize_gateway_frame(&self) -> GatewayFrame;

    /// Synthesizes `self` into both a 128-byte [`GatewayFrame`] and its register-resident 8 `u64`
    /// SWAR phase words (`[u64; 8]`), eliminating stack store-to-load forwarding stalls across
    /// downstream ensemble folding, `ActionHub` steering, and Online Toroidal $k$-Means steps.
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let frame = self.synthesize_gateway_frame();
        let words = *frame.phase_words_ref();
        (frame, words)
    }

    /// Lightweight internal/ensemble fold hook returning `(GatewayFrame, [u64; 8])` where
    /// `GatewayFrame.phase_deltas`, `mean_phase_tag_q7`, and `tagged_ptr` may be deferred until
    /// the outer `[T]` ensemble accumulator completes its final reduction.
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        self.synthesize_gateway_frame_with_words()
    }

    /// Fast predicate returning `true` if `self` carries an active non-passive signal during `[T]` ensemble folding.
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        true
    }

    /// Synthesizes a contiguous slice of `Self` into a 128-byte [`GatewayFrame`] and its 8
    /// register-resident `u64` SWAR phase words (`[u64; 8]`), accumulating scalar metadata in
    /// registers and materializing `GatewayFrame` only once upon loop completion.
    #[inline(always)]
    fn synthesize_gateway_slice_with_words(
        slice: &[Self],
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES])
    where
        Self: Sized,
    {
        fold_gateway_slice_with_query_words_core(slice, None)
    }

    /// Evaluates this signal through its live dynamic cognitive aperture against all three core
    /// layers (`Liminal` + `Stigmergic` + `Ego`) prior to final frame synthesis, gating out-of-window
    /// SWAR granules and applying in-place three-layer Ego modulation.
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let frame = self.synthesize_gateway_with_aperture_from_stack(stack);
        let modulated =
            stack.intervene_ego_and_modulate_frame_branchless(delta, horizon, venn_report, frame);
        let out_words = *modulated.phase_words_ref();
        (modulated, out_words)
    }

    /// Read-only counterpart to [`Self::synthesize_gateway_with_core_layers`] for per-layer
    /// Hot-Path RoPE steering: resolves the dynamic trait cognitive aperture and modulates `frame`
    /// across all three core layers (`Liminal` + `Stigmergic` + `Ego`) without mutating `stack`.
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let frame = self.synthesize_gateway_with_aperture_from_stack(stack);
        let modulated =
            stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, venn_report, frame);
        let out_words = *modulated.phase_words_ref();
        (modulated, out_words)
    }

    /// Shared helper resolving `self`'s dynamic cognitive aperture against `stack` and shaping
    /// the base synthesized [`GatewayFrame`].
    #[inline(always)]
    fn synthesize_gateway_with_aperture_from_stack(
        &self,
        stack: &AutonomousCoreLayerStack,
    ) -> GatewayFrame {
        let (dyn_mask, dyn_gain_q7, dyn_shift_q7) = self.resolve_dynamic_core_aperture(stack);
        let (mut frame, mut words) = self.synthesize_gateway_frame_with_words();
        apply_dynamic_aperture_to_frame_and_words(
            &mut frame,
            &mut words,
            dyn_mask,
            dyn_gain_q7,
            dyn_shift_q7,
        );
        frame
    }

    /// Dynamically routes and combines `self` (`A`) and `other` (`B`) across the 8 SWAR granules
    /// according to their live core-layer cognitive apertures:
    /// - `mask_a & !mask_b`: exclusive routing to `A`
    /// - `!mask_a & mask_b`: exclusive routing to `B`
    /// - `mask_a & mask_b`: Venn overlap cooperative fusion or Layer 2 Ego conflict tie-breaking
    /// - `!mask_a & !mask_b`: gated to `Liminal` exterior (`0`)
    #[inline]
    fn combine_pair_with_core_layers<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let combined = combine_pair_aperture_frame_pure(self, other, stack);
        let modulated = stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            venn_report,
            combined,
        );
        let final_words = *modulated.phase_words_ref();
        (modulated, final_words)
    }

    /// Read-only counterpart to [`Self::combine_pair_with_core_layers`] for per-layer
    /// Hot-Path steering without mutating `stack`.
    #[inline]
    fn combine_pair_with_core_layers_readonly<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let combined = combine_pair_aperture_frame_pure(self, other, stack);
        let modulated =
            stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, venn_report, combined);
        let final_words = *modulated.phase_words_ref();
        (modulated, final_words)
    }

    /// Aggregates a slice of fragmented signal experts (`fragments`) across the 8 SWAR granules
    /// using per-fragment dynamic core apertures, quorum agreement amplification, Layer 0
    /// uncertainty damping, and Layer 2 Ego tie-breaking (`0` heap allocations).
    #[inline]
    fn synthesize_fragment_ensemble_with_core_layers(
        fragments: &[Self],
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (
        GatewayFrame,
        [u64; NUM_STIGMERGIC_GRANULES],
        StigmergicFragmentEnsembleReport,
    )
    where
        Self: Sized,
    {
        if fragments.is_empty() {
            return (
                GatewayFrame::default(),
                [0u64; NUM_STIGMERGIC_GRANULES],
                StigmergicFragmentEnsembleReport::default(),
            );
        }

        let mut pos_votes = [0u8; NUM_STIGMERGIC_GRANULES];
        let mut neg_votes = [0u8; NUM_STIGMERGIC_GRANULES];
        let mut pos_energy = [0.0_f32; NUM_STIGMERGIC_GRANULES];
        let mut neg_energy = [0.0_f32; NUM_STIGMERGIC_GRANULES];
        let mut pos_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut neg_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut slot_votes = [0.0_f32; MAX_ATTRACTOR_DOMAINS];

        let mut active_frag_count = 0u8;
        let mut best_e = 0.0_f32;
        let mut seed_frame = fragments[0].synthesize_gateway_frame();
        let mut prov_bits = 0u32;
        let mut rep_sev = 0.0_f32;
        let mut action_flags = 0u8;

        let venn_has_info = (venn_report.venn_core_mask
            | venn_report.venn_penumbra_mask
            | venn_report.void_exterior_mask)
            != 0;
        let venn_core_boost = 1.25_f32
            + 0.25_f32
                * branchless_clamp_f32(venn_report.venn_intersection_energy * 0.25, 0.0, 1.0);

        let mut idx = 0usize;
        while idx < fragments.len() {
            let frag = &fragments[idx];
            if !frag.has_active_gateway_ensemble_signal() {
                idx += 1;
                continue;
            }
            let (frag_mask, frag_gain_q7, frag_shift_q7) =
                frag.resolve_dynamic_core_aperture(stack);
            let (f, fw) = frag.synthesize_gateway_ensemble_item();
            let raw_e = branchless_flush_subnormal_f32(branchless_relu_f32(f.effective_energy));
            let f_e = raw_e * ((f32::from(frag_gain_q7) + 32.0) * (1.0 / 128.0));
            let core_overlap_cnt = (frag_mask & venn_report.venn_core_mask).count_ones() as f32;
            let pen_overlap_cnt = (frag_mask & venn_report.venn_penumbra_mask).count_ones() as f32;
            let venn_frag_mult = if venn_has_info {
                0.50_f32 + 0.20_f32 * core_overlap_cnt + 0.05_f32 * pen_overlap_cnt
            } else {
                1.0_f32
            };
            let venn_weighted_e = f_e * venn_frag_mult;

            if f_e > DEFAULT_ENTROPY_EPS || f.provenance_bits != 0 {
                active_frag_count = active_frag_count.saturating_add(1);
                prov_bits |= f.provenance_bits;
                rep_sev = branchless_max_f32(rep_sev, f.repulsion_severity);
                action_flags |= f.action_mode_flags;
                let s_idx = usize::from(f.dominant_slot & ATTRACTOR_SLOT_ID_MASK);
                slot_votes[s_idx] += venn_weighted_e;
                if venn_weighted_e > best_e {
                    best_e = venn_weighted_e;
                    seed_frame = f;
                }

                let mut g = 0usize;
                while g < NUM_STIGMERGIC_GRANULES {
                    let bit = 1u8 << g;
                    let shaped = shape_granule_word_by_aperture(fw[g], frag_gain_q7, frag_shift_q7);
                    let in_window =
                        ((frag_mask & bit) != 0) & (f_e > DEFAULT_ENTROPY_EPS) & (shaped != 0);
                    if in_window {
                        let venn_g_weight = if !venn_has_info {
                            1.0_f32
                        } else if (venn_report.venn_core_mask & bit) != 0 {
                            venn_core_boost
                        } else if (venn_report.venn_penumbra_mask & bit) != 0 {
                            0.75_f32
                        } else if (venn_report.serendipity_excursion_mask & bit) != 0 {
                            0.90_f32
                        } else {
                            0.35_f32
                        };
                        let g_e = f_e * venn_g_weight;
                        let mean_g = swar_q7_mean_i8_from_words_64(&[shaped; 8]);
                        if mean_g >= 0 {
                            accumulate_fragment_granule_vote(
                                &mut pos_words[g],
                                &mut pos_energy[g],
                                &mut pos_votes[g],
                                shaped,
                                g_e,
                            );
                        } else {
                            accumulate_fragment_granule_vote(
                                &mut neg_words[g],
                                &mut neg_energy[g],
                                &mut neg_votes[g],
                                shaped,
                                g_e,
                            );
                        }
                    }
                    g += 1;
                }
            }
            idx += 1;
        }

        let ego_active = stack.ego_layer.is_active();
        let mut out_words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut quorum_agreed_mask = 0u8;
        let mut split_contested_mask = 0u8;
        let mut ego_tiebreak_mask = 0u8;
        let mut consensus_energy = 0.0_f32;
        let mut active_granule_mask = 0u8;

        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let bit = 1u8 << g;
            let pv = pos_votes[g];
            let nv = neg_votes[g];
            let has_any = (pv | nv) != 0;
            active_granule_mask |= branchless_select_u8(has_any, bit, 0);

            let maj_votes = pv.max(nv);
            let min_votes = pv.min(nv);
            let agreed = (maj_votes >= 2) & (maj_votes > min_votes);
            let contested = (pv > 0) & (nv > 0) & (pv == nv);
            quorum_agreed_mask |= branchless_select_u8(agreed, bit, 0);
            split_contested_mask |= branchless_select_u8(contested, bit, 0);

            let ego_can_break = ego_active & (stack.ego_layer.granule_autonomy_q7[g] >= 16);
            let ego_tie = contested & ego_can_break;
            ego_tiebreak_mask |= branchless_select_u8(ego_tie, bit, 0);

            let ego_pref_pos = stack.ego_layer.sovereign_granule_phase_q7[g] >= 0;
            let e_pref_pos = pos_energy[g] >= neg_energy[g];
            let choose_pos = branchless_select_u8(
                contested,
                branchless_select_u8(ego_can_break, u8::from(ego_pref_pos), u8::from(e_pref_pos)),
                u8::from((pv > nv) | ((pv == nv) & e_pref_pos)),
            ) != 0;

            let chosen_w = branchless_select_u64(choose_pos, pos_words[g], neg_words[g]);
            let ego_word = (stack.ego_layer.sovereign_granule_phase_q7[g] as u8 as u64)
                .wrapping_mul(SWAR_SPLAT_BYTE_64);
            let damped_contested = swar_q7_scale_le128_with_bias_x8(
                chosen_w,
                64,
                64u64.wrapping_mul(SWAR_SPLAT_BYTE_64),
            );
            let ego_resolved = swar_q7_avg_round_x8(chosen_w, ego_word);

            let gw = branchless_select_u64(
                !has_any,
                0,
                branchless_select_u64(
                    ego_tie,
                    ego_resolved,
                    branchless_select_u64(contested, damped_contested, chosen_w),
                ),
            );
            out_words[g] = gw;

            let win_e = branchless_select_f32(choose_pos, pos_energy[g], neg_energy[g]);
            let ag_mult = 1.0 + 0.12 * f32::from(maj_votes.saturating_sub(1).min(4));
            let g_contrib = branchless_select_f32(
                agreed,
                win_e * ag_mult,
                branchless_select_f32(contested & !ego_can_break, win_e * 0.5, win_e),
            );
            consensus_energy += g_contrib * (1.0 / (NUM_STIGMERGIC_GRANULES as f32));
            g += 1;
        }

        let mut dom_slot = seed_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let mut best_slot_score = -1.0_f32;
        let mut s = 0usize;
        while s < MAX_ATTRACTOR_DOMAINS {
            let mut sc = slot_votes[s];
            if ego_active {
                sc += f32::from(stack.ego_layer.slot_affinity_bias_q7[s].max(0)) * (1.0 / 64.0);
            }
            if sc > best_slot_score {
                best_slot_score = sc;
                dom_slot = s as u8;
            }
            s += 1;
        }

        let agreed_cnt = quorum_agreed_mask.count_ones() as f32;
        let diversity_gain = if active_frag_count > 1 {
            1.0 + 0.08 * agreed_cnt
        } else {
            1.0
        };

        let mut out_frame = seed_frame;
        out_frame.effective_energy =
            branchless_min_f32(consensus_energy * diversity_gain, MAX_ATTRACTOR_PULL_FORCE);
        out_frame.provenance_bits = prov_bits;
        out_frame.repulsion_severity = rep_sev;
        out_frame.dominant_slot = dom_slot;
        out_frame.action_mode_flags = branchless_select_u8(
            action_flags != 0,
            action_flags & TAG_ACTION_MODE_MASK,
            out_frame.action_mode_flags,
        );
        out_frame.active_swarm_cell_mask = active_granule_mask;
        *out_frame.phase_words_mut() = out_words;
        let tag_q7 = swar_q7_mean_i8_from_words_64(out_frame.phase_words_ref());
        let ptr_tag = pack_action_slot_tag(dom_slot, out_frame.action_mode_flags);
        out_frame.mean_phase_tag_q7 = tag_q7;
        out_frame.frame_flags = ptr_tag as u16;
        out_frame.tagged_ptr =
            gcso_tagged_ptr_t::pack(out_frame.mapped_section_offset48, ptr_tag, tag_q7).raw();

        let modulated = stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            venn_report,
            out_frame,
        );
        let final_words = *modulated.phase_words_ref();
        let report = StigmergicFragmentEnsembleReport {
            ensemble_consensus_energy: modulated.effective_energy,
            ensemble_diversity_gain: diversity_gain,
            quorum_agreed_granule_mask: quorum_agreed_mask,
            split_contested_granule_mask: split_contested_mask,
            ego_tiebreak_granule_mask: ego_tiebreak_mask,
            active_fragment_count: active_frag_count,
            dominant_fragment_slot: modulated.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            reserved: [0; 3],
        };
        (modulated, final_words, report)
    }
}

#[inline(always)]
fn accumulate_fragment_granule_vote(
    acc_word: &mut u64,
    acc_energy: &mut f32,
    acc_votes: &mut u8,
    shaped: u64,
    f_e: f32,
) {
    let new_e = *acc_energy + f_e;
    if *acc_votes == 0 {
        *acc_word = shaped;
    } else {
        let w_ratio = branchless_clamp_f32(
            f_e * fast_inv_pos_normal_f32(branchless_max_f32(new_e, DEFAULT_ENTROPY_EPS)),
            0.0,
            1.0,
        );
        let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
        let inv_s = 128u64.wrapping_sub(s_u64);
        *acc_word = swar_q7_circular_add_x8(
            swar_q7_scale_le128_with_bias_x8(
                *acc_word,
                inv_s,
                s_u64.wrapping_mul(SWAR_SPLAT_BYTE_64),
            ),
            swar_q7_scale_le128_with_bias_x8(shaped, s_u64, inv_s.wrapping_mul(SWAR_SPLAT_BYTE_64)),
        );
    }
    *acc_energy = new_e;
    *acc_votes = acc_votes.saturating_add(1);
}

#[inline(always)]
#[must_use]
fn shape_granule_word_by_aperture(word: u64, gain_q7: u8, shift_q7: i8) -> u64 {
    let shift_word = (shift_q7 as u8 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let shifted = swar_q7_circular_add_x8(word, shift_word);
    let scale128 = (u64::from(gain_q7) + 32).min(128);
    let bias = (128u64.wrapping_sub(scale128)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    swar_q7_scale_le128_with_bias_x8(shifted, scale128, bias)
}

#[inline(always)]
fn apply_dynamic_aperture_to_frame_and_words(
    frame: &mut GatewayFrame,
    words: &mut [u64; NUM_STIGMERGIC_GRANULES],
    dyn_mask: u8,
    dyn_gain_q7: u8,
    dyn_shift_q7: i8,
) {
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let open = ((dyn_mask >> g) & 1) != 0;
        let shaped = shape_granule_word_by_aperture(words[g], dyn_gain_q7, dyn_shift_q7);
        words[g] = branchless_select_u64(open, shaped, 0);
        g += 1;
    }
    let open_frac =
        f32::from(dyn_mask.count_ones() as u8) * (1.0 / (NUM_STIGMERGIC_GRANULES as f32));
    let gain_scale = (f32::from(dyn_gain_q7) + 32.0) * (1.0 / 128.0);
    frame.effective_energy = branchless_min_f32(
        branchless_relu_f32(frame.effective_energy) * open_frac * gain_scale,
        MAX_ATTRACTOR_PULL_FORCE,
    );
    frame.active_swarm_cell_mask &= dyn_mask;
    *frame.phase_words_mut() = *words;
    let tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
    let ptr_tag = pack_action_slot_tag(
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
        frame.action_mode_flags,
    );
    frame.mean_phase_tag_q7 = tag_q7;
    frame.frame_flags = ptr_tag as u16;
    frame.tagged_ptr =
        gcso_tagged_ptr_t::pack(frame.mapped_section_offset48, ptr_tag, tag_q7).raw();
}

/// Folds an arbitrary-length slice of [`ManifoldGatewaySignal`] items into a single
/// [`GatewayFrame`] and register-resident SWAR phase words, optionally weighting each item
/// by its $SO(2)^{64}$ phase coherence against `query_words_opt`.
#[inline(always)]
pub(super) fn mask_words_for_active_heads_64(
    words: &[u64; NUM_STIGMERGIC_GRANULES],
    active_heads: u16,
) -> [u64; NUM_STIGMERGIC_GRANULES] {
    if active_heads >= MAX_ATTENTION_HEADS as u16 {
        return *words;
    }
    let h = active_heads as usize;
    let mut out = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut w_idx = 0usize;
    while w_idx < NUM_STIGMERGIC_GRANULES {
        let base = w_idx * 8;
        if base + 8 <= h {
            out[w_idx] = words[w_idx];
        } else if base < h {
            let rem = (h - base) as u32;
            let mask = (1u64 << (rem * 8)) - 1;
            out[w_idx] = words[w_idx] & mask;
        }
        w_idx += 1;
    }
    out
}

/// Evaluates $SO(2)^H$ circular phase coherence restricted to the `active_heads` ($H \in [1, 64]$)
/// active attention heads, algebraically subtracting the $+1.0$ zero-padding bias contributed by
/// the $(64 - H)$ inactive SIMD/SWAR lanes.
#[inline(always)]
pub(super) fn coherence_for_active_heads_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    a_uni: bool,
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    b_uni: bool,
    active_heads: u16,
) -> f32 {
    let h = active_heads.clamp(1, MAX_ATTENTION_HEADS as u16);
    if h >= MAX_ATTENTION_HEADS as u16 || (a_uni && b_uni) {
        return coherence_from_word_arrays_with_both_uniform_flags_64(a, a_uni, b, b_uni);
    }
    let a_masked = mask_words_for_active_heads_64(a, h);
    let b_masked = mask_words_for_active_heads_64(b, h);
    let c64 =
        coherence_from_word_arrays_with_both_uniform_flags_64(&a_masked, false, &b_masked, false);
    let h_f32 = f32::from(h);
    let pad_f32 = (MAX_ATTENTION_HEADS as f32) - h_f32;
    branchless_clamp_f32(
        ((MAX_ATTENTION_HEADS as f32) * c64 - pad_f32) / h_f32,
        -1.0,
        1.0,
    )
}

/// Evaluates scale-invariant multi-head tangent cosine similarity in $[-1.0, 1.0]$ over the first
/// `active_heads` ($H \in [1, 64]$) lanes, eliminating both zero-padding bias and small-amplitude
/// circular cosine bias when comparing heterogeneous tangent steering vectors.
#[inline(always)]
pub(crate) fn tangent_cosine_active_heads_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    active_heads: u16,
) -> f32 {
    let h_usize = (active_heads.clamp(1, MAX_ATTENTION_HEADS as u16)) as usize;
    let a_bytes = words_to_q7_array_64(a);
    let b_bytes = words_to_q7_array_64(b);
    let mut dot = 0i32;
    let mut norm_a = 0i32;
    let mut norm_b = 0i32;
    let mut idx = 0usize;
    while idx < h_usize {
        let av = i32::from(a_bytes[idx]);
        let bv = i32::from(b_bytes[idx]);
        dot += av * bv;
        norm_a += av * av;
        norm_b += bv * bv;
        idx += 1;
    }
    if norm_a <= 0 || norm_b <= 0 {
        return 0.0;
    }
    let denom_sq = (norm_a as f32) * (norm_b as f32);
    let half = 0.5_f32 * denom_sq;
    let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(denom_sq.to_bits() >> 1));
    let y1 = y0 * (1.5_f32 - half * y0 * y0);
    let inv_denom = y1 * (1.5_f32 - half * y1 * y1);
    branchless_clamp_f32((dot as f32) * inv_denom, -1.0, 1.0)
}

/// Linearly interpolates two 64-lane signed Q7 tangent arrays in $[-\theta_{\mathrm{clamp}}, +\theta_{\mathrm{clamp}}]^{64}$
/// (`[-127, +127]`) without modular $\pm 128$ wrapping overflow when $|b_h - a_h| > 127$.
#[inline(always)]
pub(super) fn swar_tangent_lerp_words_with_uniform_flag_64(
    a: &[u64; NUM_STIGMERGIC_GRANULES],
    a_uni: bool,
    b: &[u64; NUM_STIGMERGIC_GRANULES],
    b_uni: bool,
    s_u64: u64,
) -> ([u64; NUM_STIGMERGIC_GRANULES], bool) {
    let s = if s_u64 > 128 { 128 } else { s_u64 };
    if s == 0 {
        return (*a, a_uni);
    }
    if s >= 128 {
        return (*b, b_uni);
    }
    let inv_s = 128u64.wrapping_sub(s);
    let bias_a = s.wrapping_mul(SWAR_SPLAT_BYTE_64);
    let bias_b = inv_s.wrapping_mul(SWAR_SPLAT_BYTE_64);
    if a_uni && b_uni {
        let a0 = swar_q7_scale_le128_with_bias_x8(a[0], inv_s, bias_a);
        let b0 = swar_q7_scale_le128_with_bias_x8(b[0], s, bias_b);
        let w0 = swar_q7_circular_add_x8(a0, b0);
        return ([w0; NUM_STIGMERGIC_GRANULES], true);
    }
    let mut out = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut k = 0usize;
    while k < NUM_STIGMERGIC_GRANULES {
        let ak = swar_q7_scale_le128_with_bias_x8(a[k], inv_s, bias_a);
        let bk = swar_q7_scale_le128_with_bias_x8(b[k], s, bias_b);
        out[k] = swar_q7_circular_add_x8(ak, bk);
        k += 1;
    }
    let is_uni = is_word_array_uniform_64(&out);
    (out, is_uni)
}

#[inline(always)]
pub(super) fn fold_gateway_slice_with_query_words_core<S: ManifoldGatewaySignal>(
    slice: &[S],
    query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    fold_gateway_slice_with_query_words_and_heads_core(
        slice,
        query_words_opt,
        MAX_ATTENTION_HEADS as u16,
    )
}

#[inline]
pub(super) fn fold_gateway_slice_with_query_words_and_heads_core<S: ManifoldGatewaySignal>(
    slice: &[S],
    query_words_opt: Option<&[u64; NUM_STIGMERGIC_GRANULES]>,
    query_active_heads: u16,
) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
    let len = slice.len();
    if len == 0 {
        return (GatewayFrame::default(), [0u64; NUM_STIGMERGIC_GRANULES]);
    }
    let use_query_resonance = query_words_opt.is_some_and(swar_words_nonzero_64);
    if len == 1 && !use_query_resonance {
        return slice[0].synthesize_gateway_frame_with_words();
    }
    let q_is_uniform = query_words_opt.is_some_and(is_word_array_uniform_64);

    let mut acc_w = [0u64; NUM_STIGMERGIC_GRANULES];
    let mut acc_is_uniform = true;
    let mut tot_e = 0.0_f32;
    let mut best_e = 0.0_f32;
    let mut has_active = false;

    let mut storage_addr = 0u64;
    let mut target_anchor_id = 0u32;
    let mut ripa_clamp_rad = DEFAULT_PHASE_CLAMP_MAX_RAD;
    let mut qdps_min_step_rad = DEFAULT_PHASE_STEP_MIN_RAD;
    let mut eff_inv_temp = 1.0_f32;
    let mut fp_coh = 1.0_f32;
    let mut rep_sev = 0.0_f32;
    let mut prov_bits = 0u32;
    let mut dom_cause = ROOT_CAUSE_NOMINAL;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut action_flags = ACTION_PRESET_EXPRESS_PHASE;
    let mut swarm_mask = 0xFFu8;
    let mut qdps_step_q7 = branchless_rad_to_phase_step_u8(DEFAULT_PHASE_STEP_MIN_RAD);
    let mut pspm_fact = 24u8;
    let mut pspm_logic = 24u8;
    let mut pspm_explore = 16u8;
    let mut fact_gain_q7 = 128u8;
    let mut logic_gain_q7 = 128u8;
    let mut explore_gain_q7 = 128u8;
    let mut active_heads = query_active_heads.clamp(1, MAX_ATTENTION_HEADS as u16);

    let mut i = 0usize;
    while i < len {
        if !slice[i].has_active_gateway_ensemble_signal() {
            i += 1;
            continue;
        }
        let (f, f_w) = slice[i].synthesize_gateway_ensemble_item();
        let raw_e = branchless_flush_subnormal_f32(branchless_relu_f32(f.effective_energy));
        let f_w_uniform = is_word_array_uniform_64(&f_w);
        let item_heads = f
            .active_head_count
            .min(query_active_heads)
            .clamp(1, MAX_ATTENTION_HEADS as u16);
        let f_e = if use_query_resonance {
            let q_w = query_words_opt.unwrap_or(&f_w);
            let coh =
                coherence_for_active_heads_64(q_w, q_is_uniform, &f_w, f_w_uniform, item_heads);
            let res = branchless_max_f32(coh, 0.0);
            branchless_flush_subnormal_f32(raw_e * (res * res + 0.02))
        } else {
            raw_e
        };
        let f_has_signal = (f_e > DEFAULT_ENTROPY_EPS)
            | (f.repulsion_severity > DEFAULT_ENTROPY_EPS)
            | (f.provenance_bits != 0)
            | ((f.action_mode_flags & ACTION_MODE_LAYER_SKIP) != 0);

        if f_has_signal {
            if !has_active {
                acc_w = f_w;
                acc_is_uniform = f_w_uniform;
                tot_e = f_e;
                best_e = f_e;
                storage_addr = f.mapped_section_offset48;
                target_anchor_id = f.target_anchor_id;
                ripa_clamp_rad = f.ripa_clamp_rad;
                qdps_min_step_rad = f.qdps_min_step_rad;
                eff_inv_temp = f.effective_inv_temp;
                fp_coh = f.fact_persona_coherence;
                rep_sev = f.repulsion_severity;
                prov_bits = f.provenance_bits;
                dom_cause = f.dominant_cause_code;
                dom_slot = f.dominant_slot;
                action_flags = f.action_mode_flags;
                swarm_mask = f.active_swarm_cell_mask;
                qdps_step_q7 = f.qdps_min_step_q7;
                pspm_fact = f.subhead_fact_heads;
                pspm_logic = f.subhead_logic_heads;
                pspm_explore = f.subhead_explore_heads;
                fact_gain_q7 = f.subhead_fact_gain_q7;
                logic_gain_q7 = f.subhead_logic_gain_q7;
                explore_gain_q7 = f.subhead_explore_gain_q7;
                active_heads = item_heads;
                has_active = true;
            } else {
                if f_e > DEFAULT_ENTROPY_EPS {
                    let new_tot = tot_e + f_e;
                    let safe_tot =
                        branchless_select_f32(new_tot > DEFAULT_ENTROPY_EPS, new_tot, 1.0);
                    let w_ratio =
                        branchless_clamp_f32(f_e * fast_inv_pos_normal_f32(safe_tot), 0.0, 1.0);
                    let s_u64 = ((w_ratio * 128.0 + 0.5) as u8).min(128) as u64;
                    if s_u64 >= 128 {
                        acc_w = f_w;
                        acc_is_uniform = f_w_uniform;
                    } else if s_u64 > 0 && !swar_words_equal_64(&acc_w, &f_w) {
                        let f_is_uniform = acc_is_uniform && f_w_uniform;
                        let bias_restore =
                            (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                        let (next_acc, next_uni) = swar_slerp_words_with_uniform_flag_64(
                            &acc_w,
                            acc_is_uniform,
                            &f_w,
                            f_is_uniform,
                            s_u64,
                            bias_restore,
                        );
                        acc_w = next_acc;
                        acc_is_uniform = next_uni;
                    }
                    eff_inv_temp = branchless_lerp_f32(eff_inv_temp, f.effective_inv_temp, w_ratio);
                    fp_coh = branchless_lerp_f32(fp_coh, f.fact_persona_coherence, w_ratio);

                    let take_dom = f_e > best_e;
                    best_e = branchless_select_f32(take_dom, f_e, best_e);
                    target_anchor_id =
                        branchless_select_u32(take_dom, f.target_anchor_id, target_anchor_id);
                    storage_addr = branchless_select_u64(
                        take_dom & (f.mapped_section_offset48 != 0),
                        f.mapped_section_offset48,
                        storage_addr,
                    );
                    dom_slot = branchless_select_u8(take_dom, f.dominant_slot, dom_slot);
                    pspm_fact = branchless_select_u8(take_dom, f.subhead_fact_heads, pspm_fact);
                    pspm_logic = branchless_select_u8(take_dom, f.subhead_logic_heads, pspm_logic);
                    pspm_explore =
                        branchless_select_u8(take_dom, f.subhead_explore_heads, pspm_explore);
                    fact_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_fact_gain_q7, fact_gain_q7);
                    logic_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_logic_gain_q7, logic_gain_q7);
                    explore_gain_q7 =
                        branchless_select_u8(take_dom, f.subhead_explore_gain_q7, explore_gain_q7);
                    tot_e = new_tot;
                }
                action_flags |= f.action_mode_flags;
                swarm_mask &= f.active_swarm_cell_mask;
                prov_bits |= f.provenance_bits;
                rep_sev = branchless_max_f32(rep_sev, f.repulsion_severity);
                if f.dominant_cause_code > dom_cause {
                    dom_cause = f.dominant_cause_code;
                }
            }
        }
        i += 1;
    }

    if !has_active {
        return slice[0].synthesize_gateway_frame_with_words();
    }

    let phase_deltas = words_to_q7_array_64(&acc_w);
    let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&acc_w, acc_is_uniform);
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = gcso_tagged_ptr_t::pack(storage_addr, packed_tag, tag_q7).raw();
    let out_energy = if use_query_resonance {
        branchless_min_f32(best_e, MAX_ATTRACTOR_PULL_FORCE)
    } else {
        branchless_min_f32(tot_e, MAX_ATTRACTOR_PULL_FORCE)
    };
    (
        GatewayFrame {
            tagged_ptr,
            mapped_section_offset48: storage_addr,
            target_anchor_id,
            effective_energy: out_energy,
            ripa_clamp_rad,
            qdps_min_step_rad,
            effective_inv_temp: eff_inv_temp,
            fact_persona_coherence: fp_coh,
            repulsion_severity: rep_sev,
            provenance_bits: prov_bits,
            dominant_cause_code: dom_cause,
            dominant_slot: dom_slot,
            action_mode_flags: action_flags,
            active_swarm_cell_mask: swarm_mask,
            qdps_min_step_q7: qdps_step_q7,
            subhead_fact_heads: pspm_fact,
            subhead_logic_heads: pspm_logic,
            subhead_explore_heads: pspm_explore,
            subhead_fact_gain_q7: fact_gain_q7,
            subhead_logic_gain_q7: logic_gain_q7,
            subhead_explore_gain_q7: explore_gain_q7,
            mean_phase_tag_q7: tag_q7,
            active_head_count: active_heads,
            frame_flags: packed_tag as u16,
            phase_deltas,
        },
        acc_w,
    )
}

impl<T: ManifoldGatewaySignal + ?Sized> ManifoldGatewaySignal for &T {
    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        (**self).synthesize_gateway_frame()
    }
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_frame_with_words()
    }
    #[inline(always)]
    fn synthesize_gateway_ensemble_item(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_ensemble_item()
    }
    #[inline(always)]
    fn has_active_gateway_ensemble_signal(&self) -> bool {
        (**self).has_active_gateway_ensemble_signal()
    }
    #[inline]
    fn synthesize_gateway_with_core_layers(
        &self,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_with_core_layers(stack, delta, horizon, venn_report)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).synthesize_gateway_with_core_layers_readonly(stack, delta, horizon, venn_report)
    }
    #[inline]
    fn combine_pair_with_core_layers<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).combine_pair_with_core_layers(other, stack, delta, horizon, venn_report)
    }
    #[inline]
    fn combine_pair_with_core_layers_readonly<B: ManifoldGatewaySignal + ?Sized>(
        &self,
        other: &B,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        (**self).combine_pair_with_core_layers_readonly(other, stack, delta, horizon, venn_report)
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
    stack: &AutonomousCoreLayerStack,
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

    let ego_active = stack.ego_layer.is_active();
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
        let ego_phase_word = (stack.ego_layer.sovereign_granule_phase_q7[g] as u8 as u64)
            .wrapping_mul(SWAR_SPLAT_BYTE_64);
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
        ripa_clamp_rad: branchless_lerp_f32(fa.ripa_clamp_rad, fb.ripa_clamp_rad, ratio_b),
        qdps_min_step_rad: branchless_min_f32(fa.qdps_min_step_rad, fb.qdps_min_step_rad),
        effective_inv_temp: branchless_lerp_f32(
            fa.effective_inv_temp,
            fb.effective_inv_temp,
            ratio_b,
        ),
        fact_persona_coherence: pair_coh,
        repulsion_severity: branchless_max_f32(fa.repulsion_severity, fb.repulsion_severity),
        provenance_bits: fa.provenance_bits | fb.provenance_bits,
        dominant_cause_code: branchless_select_u8(
            fa.dominant_cause_code >= fb.dominant_cause_code,
            fa.dominant_cause_code,
            fb.dominant_cause_code,
        ),
        dominant_slot: branchless_select_u8(dom_a, fa.dominant_slot, fb.dominant_slot),
        action_mode_flags: fa.action_mode_flags | fb.action_mode_flags,
        active_swarm_cell_mask: combined_mask
            & (fa.active_swarm_cell_mask | fb.active_swarm_cell_mask),
        qdps_min_step_q7: branchless_select_u8(
            fa.qdps_min_step_q7 <= fb.qdps_min_step_q7,
            fa.qdps_min_step_q7,
            fb.qdps_min_step_q7,
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
    combined.tagged_ptr =
        gcso_tagged_ptr_t::pack(combined.mapped_section_offset48, ptr_tag, tag_q7).raw();
    combined
}

#[inline]
fn combine_pair_aperture_frame_pure<
    A: ManifoldGatewaySignal + ?Sized,
    B: ManifoldGatewaySignal + ?Sized,
>(
    a: &A,
    b: &B,
    stack: &AutonomousCoreLayerStack,
) -> GatewayFrame {
    let ap_a = a.resolve_dynamic_core_aperture(stack);
    let ap_b = b.resolve_dynamic_core_aperture(stack);
    let (fa, wa) = a.synthesize_gateway_frame_with_words();
    let (fb, wb) = b.synthesize_gateway_frame_with_words();
    combine_pair_aperture_from_frames(fa, wa, ap_a, fb, wb, ap_b, stack)
}

#[inline]
fn combine_fact_persona_aperture_frame_pure<
    F: ObjectiveFactSignal + ?Sized,
    P: SubjectivePersonaSignal + ?Sized,
>(
    fact: &F,
    persona: &P,
    stack: &AutonomousCoreLayerStack,
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
fn apply_sys_and_guard_to_pair_frame<
    S: SystemConstraintSignal + ?Sized,
    G: ContradictionGuardSignal + ?Sized,
>(
    sys: &S,
    guard_opt: Option<&G>,
    mut frame: GatewayFrame,
) -> GatewayFrame {
    let mut prov_bits =
        frame.provenance_bits | sys.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT;
    let active_swarm_mask = frame.active_swarm_cell_mask & sys.active_swarm_cells_mask();
    let force_skip = sys.recommend_layer_skip();
    let qdps_rad = branchless_max_f32(0.0, sys.qdps_min_step_rad_override());
    let sys_sev = branchless_clamp_f32(sys.causal_anomaly_severity(), 0.0, 1.0);
    if force_skip | (sys.active_swarm_cells_mask() != 0xFF) {
        prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
    }

    let mut rep_sev = branchless_max_f32(frame.repulsion_severity, sys_sev);
    let words = *frame.phase_words_ref();
    if let Some(guard) = guard_opt {
        const ZERO_PHASES_64: [gcso_q7_t; MAX_ATTENTION_HEADS] = [0i8; MAX_ATTENTION_HEADS];
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
        let pw = frame.phase_words_mut();
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            pw[c] = swar_q7_neg_sat_x8(pw[c]);
            c += 1;
        }
        frame.phase_deltas = words_to_q7_array_64(pw);
        frame.dominant_slot = ATTRACTOR_SLOT_REPULSION;
        rep_sev = branchless_max_f32(rep_sev, 0.85);
        frame.action_mode_flags = ACTION_PRESET_REPULSION_ESCAPE;
        frame.dominant_cause_code = ROOT_CAUSE_COHOMOLOGY_REPULSION;
        frame.effective_inv_temp = 0.75;
    } else if force_skip {
        frame.action_mode_flags |= ACTION_MODE_LAYER_SKIP;
    }

    frame.provenance_bits = prov_bits;
    frame.active_swarm_cell_mask = active_swarm_mask;
    frame.qdps_min_step_rad = branchless_max_f32(frame.qdps_min_step_rad, qdps_rad);
    frame.qdps_min_step_q7 = branchless_rad_to_phase_step_u8(frame.qdps_min_step_rad);
    frame.repulsion_severity = rep_sev;
    let tag_q7 = swar_q7_mean_i8_from_words_64(frame.phase_words_ref());
    let ptr_tag = pack_action_slot_tag(
        frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
        frame.action_mode_flags,
    );
    frame.mean_phase_tag_q7 = tag_q7;
    frame.frame_flags = ptr_tag as u16;
    frame.tagged_ptr =
        gcso_tagged_ptr_t::pack(frame.mapped_section_offset48, ptr_tag, tag_q7).raw();
    frame
}

impl CausalProvenanceSignal for GatewayFrame {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.provenance_bits
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
    let mut qdps_rad = DEFAULT_PHASE_STEP_MIN_RAD;
    let mut sys_sev = 0.0_f32;

    if let Some(sys) = sys_opt {
        prov_bits |= sys.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT;
        active_swarm_mask = sys.active_swarm_cells_mask();
        force_skip = sys.recommend_layer_skip();
        qdps_rad = branchless_max_f32(0.0, sys.qdps_min_step_rad_override());
        sys_sev = branchless_clamp_f32(sys.causal_anomaly_severity(), 0.0, 1.0);
        if force_skip | (active_swarm_mask != 0xFF) {
            prov_bits |= CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE;
        }
    }
    let qdps_step_u8 = branchless_rad_to_phase_step_u8(qdps_rad);

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
        fact_addr = fa & gcso_tagged_ptr_t::ADDR_MASK;
        fact_id = fid;
        fact_heads = fh.clamp(1, MAX_ATTENTION_HEADS as u16);
        let fc = fact.fact_ripa_clamp_rad();
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
        pers_addr = pa & gcso_tagged_ptr_t::ADDR_MASK;
        pers_id = pid;
        pers_heads = ph.clamp(1, MAX_ATTENTION_HEADS as u16);
        let pc = pers.persona_ripa_clamp_rad();
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
    #[allow(unused_mut)]
    let mut avx_precomputed_tag: Option<gcso_q7_t> = None;
    let mut eff_energy = 0.0_f32;
    let mut raw_clamp = DEFAULT_PERSONA_PHASE_CLAMP_RAD;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut dom_anchor_id = 0u32;
    let mut storage_addr = 0u64;
    let mut pspm_fact = 24u8;
    let mut pspm_logic = 24u8;
    let mut pspm_explore = 16u8;

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
        pspm_fact = branchless_select_u8(fact_dom, 32, 20);
        pspm_logic = 24;
        pspm_explore = branchless_select_u8(fact_dom, 8, 20);
    } else if has_fact {
        blended_w = fact_w;
        blended_uni = is_word_array_uniform_64(&fact_w);
        eff_energy = branchless_min_f32(fact_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = fact_clamp;
        dom_slot = ATTRACTOR_SLOT_KNOWLEDGE;
        dom_anchor_id = fact_id;
        storage_addr = fact_addr;
        pspm_fact = 36;
        pspm_logic = 20;
        pspm_explore = 8;
    } else if has_persona {
        blended_w = pers_w;
        blended_uni = is_word_array_uniform_64(&pers_w);
        eff_energy = branchless_min_f32(pers_e, MAX_ATTRACTOR_PULL_FORCE);
        raw_clamp = pers_clamp;
        dom_slot = ATTRACTOR_SLOT_PERSONA;
        dom_anchor_id = pers_id;
        storage_addr = pers_addr;
        pspm_fact = 16;
        pspm_logic = 24;
        pspm_explore = 24;
    }

    let norm_ripa_clamp = {
        if raw_clamp < DEFAULT_FACT_PHASE_CLAMP_RAD {
            let auth_ratio = branchless_clamp_f32(raw_clamp * INV_FACT_RIPA_CLAMP_RAD, 0.10, 1.0);
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

    if qdps_step_u8 > 1 {
        let (fw, fu) = swar_q7_phase_step_filter_words_with_uniform_flag_64(
            &blended_w,
            blended_uni,
            qdps_step_u8,
        );
        blended_w = fw;
        blended_uni = fu;
    }

    let mut rep_sev = 0.0_f32;
    if let Some(guard) = guard_opt {
        const ZERO_PHASES_64: [gcso_q7_t; MAX_ATTENTION_HEADS] = [0i8; MAX_ATTENTION_HEADS];
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
        if blended_uni {
            blended_w = [swar_q7_neg_sat_x8(blended_w[0]); NUM_STIGMERGIC_GRANULES];
        } else {
            blended_w = [
                swar_q7_neg_sat_x8(blended_w[0]),
                swar_q7_neg_sat_x8(blended_w[1]),
                swar_q7_neg_sat_x8(blended_w[2]),
                swar_q7_neg_sat_x8(blended_w[3]),
                swar_q7_neg_sat_x8(blended_w[4]),
                swar_q7_neg_sat_x8(blended_w[5]),
                swar_q7_neg_sat_x8(blended_w[6]),
                swar_q7_neg_sat_x8(blended_w[7]),
            ];
        }
        dom_slot = ATTRACTOR_SLOT_REPULSION;
        rep_sev = branchless_max_f32(rep_sev, 0.85);
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
        let (v0, v1) = unsafe { crate::foundation::branchless::avx2_words64_to_vec64(&blended_w) };
        avx_precomputed_tag =
            Some(unsafe { crate::foundation::avx2_q7_mean_i8_from_vec64(v0, v1) });
    }
    let tag_q7 = match avx_precomputed_tag {
        Some(t) => t,
        None => swar_q7_mean_i8_from_words_with_uniform_flag_64(&blended_w, blended_uni),
    };
    let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
    let tagged_ptr = gcso_tagged_ptr_t::pack(storage_addr, packed_tag, tag_q7).raw();
    let logic_gain_q7 = ((fact_gain_q7 as u16 + pers_gain_q7 as u16) >> 1) as u8;

    (
        GatewayFrame {
            tagged_ptr,
            mapped_section_offset48: storage_addr,
            target_anchor_id: dom_anchor_id,
            effective_energy: eff_energy,
            ripa_clamp_rad: norm_ripa_clamp,
            qdps_min_step_rad: qdps_rad,
            effective_inv_temp: inv_temp,
            fact_persona_coherence: fp_coh,
            repulsion_severity: branchless_max_f32(rep_sev, sys_sev),
            provenance_bits: prov_bits,
            dominant_cause_code: dominant_cause,
            dominant_slot: dom_slot,
            action_mode_flags: action_flags,
            active_swarm_cell_mask: active_swarm_mask,
            qdps_min_step_q7: qdps_step_u8,
            subhead_fact_heads: pspm_fact,
            subhead_logic_heads: pspm_logic,
            subhead_explore_heads: pspm_explore,
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

// ===================================================================
// 7. Blanket `ManifoldGatewaySignal` Implementations for Wrappers, Tuples & Ensembles
// ===================================================================

impl<T: BranchlessKernel> ManifoldGatewaySignal for SystemConstraint<T> {
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

impl ManifoldGatewaySignal for gcso_edbc_state_t {
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

impl ManifoldGatewaySignal for gcso_config_t {
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

impl<B: BranchlessKernel> ManifoldGatewaySignal for ContradictionGuard<B>
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
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let combined = combine_fact_persona_aperture_frame_pure(&self.0, &self.1, stack);
        let modulated = stack.intervene_ego_and_modulate_frame_branchless(
            delta,
            horizon,
            venn_report,
            combined,
        );
        let words = *modulated.phase_words_ref();
        (modulated, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let combined = combine_fact_persona_aperture_frame_pure(&self.0, &self.1, stack);
        let modulated =
            stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, venn_report, combined);
        let words = *modulated.phase_words_ref();
        (modulated, words)
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
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let guarded = apply_sys_and_guard_to_pair_frame::<S, ContradictionGuard<()>>(
            &self.0, None, pair_frame,
        );
        let modulated =
            stack.intervene_ego_and_modulate_frame_branchless(delta, horizon, venn_report, guarded);
        let words = *modulated.phase_words_ref();
        (modulated, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let guarded = apply_sys_and_guard_to_pair_frame::<S, ContradictionGuard<()>>(
            &self.0, None, pair_frame,
        );
        let modulated =
            stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, venn_report, guarded);
        let words = *modulated.phase_words_ref();
        (modulated, words)
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
        stack: &mut AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let guarded = apply_sys_and_guard_to_pair_frame(&self.0, Some(&self.3), pair_frame);
        let modulated =
            stack.intervene_ego_and_modulate_frame_branchless(delta, horizon, venn_report, guarded);
        let words = *modulated.phase_words_ref();
        (modulated, words)
    }
    #[inline]
    fn synthesize_gateway_with_core_layers_readonly(
        &self,
        stack: &AutonomousCoreLayerStack,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        venn_report: &StigmergicOverlapSerendipityReport,
    ) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
        let pair_frame = combine_fact_persona_aperture_frame_pure(&self.1, &self.2, stack);
        let guarded = apply_sys_and_guard_to_pair_frame(&self.0, Some(&self.3), pair_frame);
        let modulated =
            stack.modulate_gateway_frame_with_ego_readonly(delta, horizon, venn_report, guarded);
        let words = *modulated.phase_words_ref();
        (modulated, words)
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
    pub fn from_q7_probe(signals: &'a [S], probe_q7: &[gcso_q7_t]) -> Self {
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
    pub fn from_trail(signals: &'a [S], trail: &gcso_pointer_trail_t) -> Self {
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
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let candidate_words = candidate_phases.project_to_words_64();
        self.evaluate_repulsion_severity_from_words(&candidate_words, candidate_phases)
    }

    #[inline]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
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
}

impl RawPhaseProjectable for GatewayFrame {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }
    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        *self.phase_words_ref()
    }
}
