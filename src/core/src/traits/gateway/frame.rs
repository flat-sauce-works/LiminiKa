// name: src/core/src/traits/gateway/frame.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single composite [GatewayFrame] (128B Pod), [ManifoldGatewaySignal], multi-origin synthesis, and ensemble folding.

use super::*;
#[allow(unused_imports)]
use super::signals::*;

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
            ripa_clamp_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
            effective_inv_temp: 1.0,
            fact_persona_coherence: 1.0,
            repulsion_severity: 0.0,
            provenance_bits: 0,
            dominant_cause_code: ROOT_CAUSE_NOMINAL,
            dominant_slot: ATTRACTOR_SLOT_SYSTEM_BASELINE,
            action_mode_flags: ACTION_PRESET_EXPRESS_PHASE,
            active_swarm_cell_mask: 0xFF,
            qdps_min_step_q7: branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD),
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
    pub const fn has_unit_pspm_gains(&self) -> bool {
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
        let (sin_t, cos_t) = branchless_sincos_q7_ripa(phase_q7, self.ripa_clamp_rad);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            // SAFETY: `target_feature = "avx2"` is enabled and `q_block32`, `k_block32`, `v_block32` are disjoint 32-float arrays.
            unsafe {
                use core::arch::x86_64::_mm256_set1_ps;
                crate::traits::branchless::avx2_rotate_so2_triple_block32(
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
                    crate::traits::primitives::branchless_rotate_so2_pair(
                        q_block32[idx],
                        q_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
                q_block32[idx] = q0;
                q_block32[idx + 1] = q1;
                let (k0, k1) =
                    crate::traits::primitives::branchless_rotate_so2_pair(
                        k_block32[idx],
                        k_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
                k_block32[idx] = k0;
                k_block32[idx + 1] = k1;
                let (v0, v1) =
                    crate::traits::primitives::branchless_rotate_so2_pair(
                        v_block32[idx],
                        v_block32[idx + 1],
                        sin_t,
                        cos_t,
                    );
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
    const GATEWAY_APERTURE_MASK: u32 = APERTURE_MANIFOLD_GATEWAY;

    /// Complement bitmask outside the Gateway horizon (`!APERTURE_MANIFOLD_GATEWAY`).
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    let c64 = coherence_from_word_arrays_with_both_uniform_flags_64(
        &a_masked, false, &b_masked, false,
    );
    let h_f32 = f32::from(h);
    let pad_f32 = (MAX_ATTENTION_HEADS as f32) - h_f32;
    branchless_clamp_f32(((MAX_ATTENTION_HEADS as f32) * c64 - pad_f32) / h_f32, -1.0, 1.0)
}

/// Evaluates scale-invariant multi-head tangent cosine similarity in $[-1.0, 1.0]$ over the first
/// `active_heads` ($H \in [1, 64]$) lanes, eliminating both zero-padding bias and small-amplitude
/// circular cosine bias when comparing heterogeneous tangent steering vectors.
#[inline(always)]
pub(super) fn tangent_cosine_active_heads_64(
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
    let mut ripa_clamp_rad = DEFAULT_RIPA_CLAMP_MAX_RAD;
    let mut qdps_min_step_rad = DEFAULT_QDPS_MIN_STEP_RAD;
    let mut eff_inv_temp = 1.0_f32;
    let mut fp_coh = 1.0_f32;
    let mut rep_sev = 0.0_f32;
    let mut prov_bits = 0u32;
    let mut dom_cause = ROOT_CAUSE_NOMINAL;
    let mut dom_slot = ATTRACTOR_SLOT_SYSTEM_BASELINE;
    let mut action_flags = ACTION_PRESET_EXPRESS_PHASE;
    let mut swarm_mask = 0xFFu8;
    let mut qdps_step_q7 = branchless_rad_to_qdps_step_u8(DEFAULT_QDPS_MIN_STEP_RAD);
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
            let coh = coherence_for_active_heads_64(
                q_w,
                q_is_uniform,
                &f_w,
                f_w_uniform,
                item_heads,
            );
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
                    eff_inv_temp =
                        branchless_lerp_f32(eff_inv_temp, f.effective_inv_temp, w_ratio);
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    let mut qdps_rad = DEFAULT_QDPS_MIN_STEP_RAD;
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
    let qdps_step_u8 = branchless_rad_to_qdps_step_u8(qdps_rad);

    let mut has_fact = false;
    let mut fact_e = 0.0_f32;
    let mut fact_addr = 0u64;
    let mut fact_id = 0u32;
    let mut fact_clamp = DEFAULT_FACT_RIPA_CLAMP_RAD;
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
            DEFAULT_FACT_RIPA_CLAMP_RAD,
        );
        let fg = branchless_clamp_f32(fact.fact_pspm_gain(), 0.25, 1.99);
        fact_gain_q7 = (fg * 128.0 + 0.5) as u8;
        if fact_e < 0.25 {
            prov_bits |= CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK;
        }
    }

    let mut has_persona = false;
    let mut pers_e = 0.0_f32;
    let mut pers_addr = 0u64;
    let mut pers_id = 0u32;
    let mut pers_clamp = DEFAULT_PERSONA_RIPA_CLAMP_RAD;
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
            DEFAULT_PERSONA_RIPA_CLAMP_RAD,
        );
        let pg = branchless_clamp_f32(pers.persona_pspm_gain(), 0.25, 1.99);
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
    let mut raw_clamp = DEFAULT_PERSONA_RIPA_CLAMP_RAD;
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
        if raw_clamp < DEFAULT_FACT_RIPA_CLAMP_RAD {
            let auth_ratio = branchless_clamp_f32(raw_clamp * INV_FACT_RIPA_CLAMP_RAD, 0.10, 1.0);
            let auth_u64 = ((auth_ratio * 128.0 + 0.5) as u8).min(128) as u64;
            if auth_u64 < 128 {
                let bias_auth = (128u64.wrapping_sub(auth_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                if blended_uni {
                    blended_w = [swar_q7_scale_le128_with_bias_x8(
                        blended_w[0],
                        auth_u64,
                        bias_auth,
                    ); NUM_STIGMERGIC_GRANULES];
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
        DEFAULT_RIPA_CLAMP_MAX_RAD
    };

    if qdps_step_u8 > 1 {
        let (fw, fu) = swar_q7_qdps_filter_words_with_uniform_flag_64(
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
        ACTION_MODE_DPSR_STEER | ACTION_MODE_SUBHEAD_ROUTE
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
        let (v0, v1) = unsafe { crate::traits::branchless::avx2_words64_to_vec64(&blended_w) };
        avx_precomputed_tag =
            Some(unsafe { crate::traits::primitives::avx2_q7_mean_i8_from_vec64(v0, v1) });
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(self.0.causal_anomaly_severity(), self.1.causal_anomaly_severity())
    }
}

impl<F: ObjectiveFactSignal, P: SubjectivePersonaSignal> ManifoldGatewaySignal for (F, P) {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
}

impl<S: SystemConstraintSignal, F: ObjectiveFactSignal, P: SubjectivePersonaSignal>
    CausalProvenanceSignal for (S, F, P)
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits() | CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | CAUSAL_AXIS_OBJECTIVE_FACT | CAUSAL_AXIS_SUBJECTIVE_PERSONA
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
            | self.2.causal_provenance_bits() | self.3.causal_provenance_bits()
            | CAUSAL_AXIS_SYSTEM_CONSTRAINT | CAUSAL_AXIS_OBJECTIVE_FACT
            | CAUSAL_AXIS_SUBJECTIVE_PERSONA | CAUSAL_AXIS_CONTRADICTION_GUARD
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_STIGMERGIC_GRANULES]) {
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
    fn synthesize_grounding_envelope(&self) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
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
            let sig_sev = sig.evaluate_repulsion_severity_from_words(candidate_words, candidate_phases);
            max_sev = branchless_max_f32(max_sev, sig_sev);
            let sig_words = sig.project_grounding_words_64();
            let coh = if is_subspace {
                tangent_cosine_active_heads_64(&self.query_phase_words, &sig_words, self.active_heads)
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

