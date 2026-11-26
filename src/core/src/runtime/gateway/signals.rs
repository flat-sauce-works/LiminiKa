// name: src/core/src/runtime/gateway/signals.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Two-Origin Signal Supertrait Hierarchy (`CausalProvenanceSignal`,
//! [`SystemConstraintSignal`], [`CognitiveGroundingSignal`], [`ObjectiveFactSignal`],
//! [`SubjectivePersonaSignal`], [`ContradictionGuardSignal`]) and [`CausalProvenanceReport`].

use super::*;
use crate::drivers::EntropyState;
use crate::foundation::{
    DEFAULT_PHASE_STEP_MIN_RAD, Pod, RawPhaseProjectable, TaggedPtr, branchless_clamp_f32,
    branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32,
    branchless_max_f32, branchless_relu_f32, fast_inv_pos_normal_f32, load_q7_words_64,
};
use crate::liminal::{
    MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES, PhaseQ7, RuntimeConfig,
    THRESHOLD_CAUSAL_PROVENANCE, THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_OBJECTIVE_FACT,
    THRESHOLD_SUBJECTIVE_PERSONA, THRESHOLD_SYSTEM_CONSTRAINT, branchless_select_f32,
    branchless_select_i32, branchless_select_u8, branchless_select_u32,
};
use crate::runtime::context::CoreStateStack;
use crate::stigmergy::{
    CognitiveTrajectoryLedger, ContrastivePhaseBloomFilter, DEFAULT_FACT_PHASE_CLAMP_RAD,
    DEFAULT_PERSONA_PHASE_CLAMP_RAD, PointerTrail, StigmergicScaleGranularity,
    StigmergicThresholds,
};

pub(super) const INV_FACT_PHASE_CLAMP_RAD: f32 = 1.0 / DEFAULT_FACT_PHASE_CLAMP_RAD;

/// Provenance axis bit (`bit 0`): Origin 1 (`SystemConstraintSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SYSTEM_CONSTRAINT: u32 = 1 << 0;
/// Provenance axis bit (`bit 1`): Origin 2-A (`ObjectiveFactSignal`) contributed to this frame.
pub const CAUSAL_AXIS_OBJECTIVE_FACT: u32 = 1 << 1;
/// Provenance axis bit (`bit 2`): Origin 2-B (`SubjectivePersonaSignal`) contributed to this frame.
pub const CAUSAL_AXIS_SUBJECTIVE_PERSONA: u32 = 1 << 2;
/// Provenance axis bit (`bit 3`): Origin 2-C (`ContradictionGuardSignal`) contributed to this frame.
pub const CAUSAL_AXIS_CONTRADICTION_GUARD: u32 = 1 << 3;

/// Provenance cause bit (`bit 8`): System QoS or low-entropy inertial throttle active.
pub const CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE: u32 = 1 << 8;
/// Provenance cause bit (`bit 9`): Objective fact retrieval energy was weak or absent.
pub const CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK: u32 = 1 << 9;
/// Provenance cause bit (`bit 10`): Subjective persona collided with objective fact ($\mathcal{C}_{\mathrm{FP}} < -\tau$).
pub const CAUSAL_CAUSE_PERSONA_FACT_COLLISION: u32 = 1 << 10;
/// Provenance cause bit (`bit 11`): Cohomological obstruction ($H^1 \neq 0$) or Bloom repulsion triggered.
pub const CAUSAL_CAUSE_COHOMOLOGY_REPULSION: u32 = 1 << 11;
/// Provenance cause bit (`bit 12`): Attractor sample count & coherence reached `.lmnk` crystallization maturity.
pub const CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED: u32 = 1 << 12;

/// Root-cause diagnosis code `0`: Nominal synergistic orchestration across active origins.
pub const ROOT_CAUSE_NOMINAL: u8 = 0;
/// Root-cause diagnosis code `1`: System QoS / VRAM pressure / inertial layer-skip throttle.
pub const ROOT_CAUSE_SYSTEM_QOS_THROTTLE: u8 = 1;
/// Root-cause diagnosis code `2`: Objective fact retrieval miss or weak grounding energy.
pub const ROOT_CAUSE_FACT_RETRIEVAL_MISS: u8 = 2;
/// Root-cause diagnosis code `3`: Destructive phase collision between Persona and Objective Fact.
pub const ROOT_CAUSE_PERSONA_FACT_COLLISION: u8 = 3;
/// Root-cause diagnosis code `4`: Cohomological obstruction ($H^1 \neq 0$) or Bloom repetition escape.
pub const ROOT_CAUSE_COHOMOLOGY_REPULSION: u8 = 4;

const _: () = {
    assert!(ROOT_CAUSE_NOMINAL == crate::stigmergy::ANCHOR_ATTRIBUTION_NOMINAL);
    assert!(CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED == crate::stigmergy::ANCHOR_LINEAGE_MATURE_BIT);
};

// ===================================================================
// 3. Root Supertrait (`CausalProvenanceSignal`) & 16-Byte Report
// ===================================================================

/// 16-byte (`#[repr(C, align(16))]`) [`Pod`] diagnostic report enabling $\mathcal{O}(1)$
/// root-cause ("Why") isolation across System Constraints, Objective Facts, Subjective Persona,
/// and Contradiction Guards.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CausalProvenanceReport {
    /// Bitwise OR of active origin axes (`CAUSAL_AXIS_*`) and diagnostic causes (`CAUSAL_CAUSE_*`).
    pub lineage_flags: u32,
    /// Cosine phase coherence $\mathcal{C}_{\mathrm{FP}} \in [-1, 1]$ between Objective Fact and Subjective Persona.
    pub fact_persona_coherence: f32,
    /// Peak anomaly or repulsion severity in $[0, 1]$ across all contributing signals.
    pub anomaly_severity: f32,
    /// Primary root-cause classification code (`ROOT_CAUSE_*`).
    pub attribution_tag: u8,
    /// Dominant 3-bit working lane ID (`ATTRACTOR_SLOT_*`, `0..=7`).
    pub dominant_slot: u8,
    /// Resolved 5-bit `ACTION_MODE_*` execution flags.
    pub action_mode_flags: u8,
    /// Active 8-cell Micro-Swarm mask (`0x00..=0xFF`).
    pub active_swarm_mask: u8,
}

// SAFETY: `CausalProvenanceReport` is `#[repr(C, align(16))]` with 16 bytes of primitive fields and zero padding.
unsafe impl Pod for CausalProvenanceReport {}

impl CausalProvenanceReport {
    /// Returns `true` if all bits in `flag_mask` are set in `self.lineage_flags`.
    #[inline(always)]
    #[must_use]
    pub const fn has_flag(&self, flag_mask: u32) -> bool {
        (self.lineage_flags & flag_mask) == flag_mask
    }

    /// Returns `true` when a destructive phase collision between Objective Fact and Subjective Persona was detected.
    #[inline(always)]
    #[must_use]
    pub const fn is_persona_fact_collision(&self) -> bool {
        (self.lineage_flags & CAUSAL_CAUSE_PERSONA_FACT_COLLISION) != 0
    }

    /// Returns `true` when cohomological obstruction ($H^1 \neq 0$) or Bloom repetition repulsion was triggered.
    #[inline(always)]
    #[must_use]
    pub const fn is_cohomology_repulsion(&self) -> bool {
        (self.lineage_flags & CAUSAL_CAUSE_COHOMOLOGY_REPULSION) != 0
    }
}

/// Root causal signal supertrait defining a finite 4-axis causal window
/// for all signals entering the LiminiKa Manifold Gateway.
pub trait CausalProvenanceSignal {
    /// Static causal provenance cognitive aperture across system and grounding origins (`0x0000_000F`).
    const CAUSAL_APERTURE_MASK: u32 = THRESHOLD_CAUSAL_PROVENANCE;

    /// Type-level complement identifying axes outside this causal window as unknowable (`!APERTURE`).
    const CAUSAL_UNOBSERVED_LACUNA_MASK: u32 = !Self::CAUSAL_APERTURE_MASK;

    /// Returns the bitwise provenance flags (`CAUSAL_AXIS_* | CAUSAL_CAUSE_*`) contributed by `self`.
    fn causal_provenance_bits(&self) -> u32;

    /// Returns the active unknowable complement bitmask derived strictly as `!self.causal_provenance_bits()`.
    #[inline(always)]
    fn active_unobserved_lacuna_bits(&self) -> u32 {
        !self.causal_provenance_bits()
    }

    /// Returns the normalized anomaly or constraint severity in $[0.0, 1.0]$.
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        0.0
    }

    /// Branchlessly resolves this signal's live dynamic cognitive aperture
    /// `(active_granule_mask, effective_gain_q7, phase_shift_q7)` across all three sheets of
    /// [`CoreStateStack`] (`Layer 0: Liminal`, `Layer 1: Stigmergic`, `Layer 2: Ego`).
    #[inline]
    fn resolve_dynamic_core_aperture(&self, stack: &CoreStateStack) -> (u8, u8, i8) {
        let bits = self.causal_provenance_bits();
        let ego_active = stack.ego.is_active();
        let ego_trans_pos = ego_active & (stack.ego.transgression_bias_q7 > 0);
        let mut active_granule_mask = 0u8;
        let mut weighted_gain_sum = 0u32;
        let mut weighted_shift_sum = 0i32;
        let mut active_axis_count = 0u32;

        let mut ch = 0usize;
        while ch < 4 {
            let axis_on = ((bits >> ch) & 1) != 0;
            let base_floor = i32::from(stack.liminal.threshold_floor_q7[ch]);
            let base_gain = i32::from(stack.liminal.aperture_gain_q7[ch]);
            let ego_floor_bias =
                branchless_select_i32(ego_active, i32::from(stack.ego.trait_floor_bias_q7[ch]), 0);
            let ego_gain_bias =
                branchless_select_i32(ego_active, i32::from(stack.ego.trait_gain_bias_q7[ch]), 0);
            let eff_floor = (base_floor + ego_floor_bias).clamp(4, 127) as u8;
            let eff_gain = (base_gain + ego_gain_bias).clamp(16, 127) as u8;
            let ap_open = ((stack.liminal.active_aperture_mask >> ch) & 1) != 0;

            let mut ch_mask = 0u8;
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                let bit = 1u8 << g;
                let pher = stack.stigmergy.granule_pheromone_q7[g];
                let in_overlap_core = (stack.stigmergy.overlap_mask & bit) != 0;
                let in_discovery = (stack.stigmergy.discovery_excursion_mask & bit) != 0;
                let ego_auto = stack.ego.granule_autonomy_q7[g];

                let overlap_bonus = branchless_select_u8(in_overlap_core, 24, 0);
                let ch_accepts_ego_transgress = (ch == 2) | (stack.ego.trait_floor_bias_q7[ch] < 0);
                let ego_bonus = branchless_select_u8(
                    ego_trans_pos & ch_accepts_ego_transgress & (ego_auto >= 24),
                    ego_auto >> 1,
                    0,
                );
                let eff_density = pher.saturating_add(overlap_bonus).saturating_add(ego_bonus);

                let req_floor = branchless_select_u8(
                    ch == 2,
                    (eff_floor >> 1).saturating_sub(branchless_select_u8(in_discovery, 8, 0)),
                    eff_floor >> 1,
                );
                let g_open = ap_open & (eff_density >= req_floor);
                ch_mask |= branchless_select_u8(g_open, bit, 0);
                g += 1;
            }

            active_granule_mask |= branchless_select_u8(axis_on, ch_mask, 0);
            weighted_gain_sum += branchless_select_u32(axis_on, u32::from(eff_gain), 0);
            weighted_shift_sum +=
                branchless_select_i32(axis_on, i32::from(stack.liminal.phase_shift_q7[ch]), 0);
            active_axis_count += u32::from(axis_on);
            ch += 1;
        }

        let unattributed = active_axis_count == 0;
        let has_layer1_mask =
            (stack.stigmergy.overlap_mask | stack.stigmergy.discovery_excursion_mask) != 0;
        let fallback_mask = stack.stigmergy.overlap_mask
            | stack.stigmergy.discovery_excursion_mask
            | branchless_select_u8(ego_trans_pos | !has_layer1_mask, 0xFF, 0);
        let out_mask = branchless_select_u8(unattributed, fallback_mask, active_granule_mask);
        let out_gain = match active_axis_count {
            0 => stack.liminal.aperture_gain_q7[7],
            1 => weighted_gain_sum.min(127) as u8,
            2 => (weighted_gain_sum >> 1).min(127) as u8,
            3 => ((weighted_gain_sum * 43) >> 7).min(127) as u8,
            _ => (weighted_gain_sum >> 2).min(127) as u8,
        };
        let out_shift = match active_axis_count {
            0 => stack.liminal.phase_shift_q7[7],
            1 => weighted_shift_sum.clamp(-127, 127) as i8,
            2 => (weighted_shift_sum >> 1).clamp(-127, 127) as i8,
            3 => ((weighted_shift_sum * 43) >> 7).clamp(-127, 127) as i8,
            _ => (weighted_shift_sum >> 2).clamp(-127, 127) as i8,
        };
        (out_mask, out_gain.max(16), out_shift)
    }
}

impl<T: CausalProvenanceSignal + ?Sized> CausalProvenanceSignal for &T {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        (**self).causal_provenance_bits()
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        (**self).causal_anomaly_severity()
    }
    #[inline(always)]
    fn resolve_dynamic_core_aperture(&self, stack: &CoreStateStack) -> (u8, u8, i8) {
        (**self).resolve_dynamic_core_aperture(stack)
    }
}

// ===================================================================
// 4. Origin 1: System & Physical Constraint Blanket (`SystemConstraintSignal`)
// ===================================================================

/// **Origin 1 Supertrait (`SystemConstraintSignal: CausalProvenanceSignal`)**:
/// Hardware resource, game-coexistence QoS, and `EntropyController` entropy constraints governing compute budgeting.
pub trait SystemConstraintSignal: CausalProvenanceSignal {
    /// Static cognitive aperture bit isolating Origin 1 system & QoS telemetry (`1 << 0`).
    const SYSTEM_APERTURE_MASK: u32 = THRESHOLD_SYSTEM_CONSTRAINT;

    /// Complement bitmask outside Origin 1 (`!THRESHOLD_SYSTEM_CONSTRAINT`).
    const SYSTEM_UNOBSERVED_LACUNA_MASK: u32 = !Self::SYSTEM_APERTURE_MASK;

    /// Normalized VRAM / system memory pressure in $[0.0, 1.0]$.
    fn system_vram_pressure(&self) -> f32;

    /// Normalized GPU/CPU frame-time overrun ratio in $[0.0, 1.0]$ for AAA game coexistence.
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        0.0
    }

    /// 8-bit bitmask of active 8-head SWAR Micro-Swarm cells (`0xFF` = all 64 heads active).
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let pressure = branchless_max_f32(
            branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0),
        );
        branchless_select_u8(
            pressure > 0.85,
            0x0F,
            branchless_select_u8(pressure > 0.65, 0x3F, 0xFF),
        )
    }

    /// Returns `true` when inertial layer skipping (`ACTION_MODE_LAYER_SKIP`) is recommended.
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        let overrun = branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0);
        let vram = branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0);
        (overrun > 0.80) | (vram > 0.90)
    }

    /// Recommended `PhaseStepFilter` minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        let pressure = branchless_max_f32(
            branchless_clamp_f32(self.system_vram_pressure(), 0.0, 1.0),
            branchless_clamp_f32(self.system_gpu_frame_overrun(), 0.0, 1.0),
        );
        DEFAULT_PHASE_STEP_MIN_RAD * (1.0 + 2.0 * pressure)
    }
}

impl<T: SystemConstraintSignal + ?Sized> SystemConstraintSignal for &T {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        (**self).system_vram_pressure()
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        (**self).system_gpu_frame_overrun()
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        (**self).active_swarm_cells_mask()
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        (**self).recommend_layer_skip()
    }
    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        (**self).phase_step_min_rad_override()
    }
}

/// Zero-cost wrapper classifying any system/resource state `T` into **Origin 1 (`SystemConstraintSignal`)**.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemConstraint<T = ()> {
    /// Wrapped inner state or telemetry payload.
    pub inner: T,
    /// Normalized VRAM / memory pressure in $[0.0, 1.0]$.
    pub vram_pressure: f32,
    /// Normalized GPU/CPU frame overrun in $[0.0, 1.0]$.
    pub gpu_frame_overrun: f32,
    /// Explicit `PhaseStepFilter` minimum step cutoff in radians.
    pub phase_step_min_rad: f32,
    /// Active 8-cell Micro-Swarm bitmask (`0x00..=0xFF`).
    pub active_swarm_mask: u8,
    /// Explicit inertial layer-skip recommendation flag.
    pub force_layer_skip: bool,
}

impl<T> SystemConstraint<T> {
    /// Constructs a new [`SystemConstraint`] wrapper with explicit QoS parameters.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        inner: T,
        vram_pressure: f32,
        gpu_frame_overrun: f32,
        active_swarm_mask: u8,
    ) -> Self {
        let p_vram = branchless_clamp_f32(branchless_relu_f32(vram_pressure), 0.0, 1.0);
        let p_gpu = branchless_clamp_f32(branchless_relu_f32(gpu_frame_overrun), 0.0, 1.0);
        let max_p = branchless_max_f32(p_vram, p_gpu);
        Self {
            inner,
            vram_pressure: p_vram,
            gpu_frame_overrun: p_gpu,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD * (1.0 + 2.0 * max_p),
            active_swarm_mask: branchless_select_u8(
                active_swarm_mask != 0,
                active_swarm_mask,
                0xFF,
            ),
            force_layer_skip: (p_gpu > 0.80) | (p_vram > 0.90),
        }
    }

    /// Builder method overriding `force_layer_skip`.
    #[inline(always)]
    #[must_use]
    pub const fn with_layer_skip(mut self, skip: bool) -> Self {
        self.force_layer_skip = skip;
        self
    }

    /// Builder method overriding `phase_step_min_rad`.
    #[inline(always)]
    #[must_use]
    pub const fn with_phase_step_min_rad(mut self, step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(step_rad) & (step_rad >= 0.0);
        self.phase_step_min_rad = branchless_select_f32(valid, step_rad, self.phase_step_min_rad);
        self
    }
}

impl<T> CausalProvenanceSignal for SystemConstraint<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let throttle = self.force_layer_skip
            | (self.active_swarm_mask != 0xFF)
            | (self.vram_pressure > 0.70)
            | (self.gpu_frame_overrun > 0.70);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(throttle, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(self.vram_pressure, self.gpu_frame_overrun)
    }
}

impl<T> SystemConstraintSignal for SystemConstraint<T> {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        self.vram_pressure
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        self.gpu_frame_overrun
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        self.active_swarm_mask
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        self.force_layer_skip
    }
    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        self.phase_step_min_rad
    }
}

impl CausalProvenanceSignal for EntropyState {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let is_rep = self.active_branch_mode == 2;
        let is_low_e = (self.active_branch_mode == 0) & (self.moving_z_entropy < 0.12);
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
            | branchless_select_u32(is_rep, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
            | branchless_select_u32(is_low_e, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let h_sev =
            branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0);
        branchless_select_f32(self.active_branch_mode == 2, 0.85, h_sev)
    }
}

impl SystemConstraintSignal for EntropyState {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        branchless_clamp_f32(branchless_relu_f32(self.moving_z_entropy) * 0.25, 0.0, 1.0)
    }
    #[inline(always)]
    fn system_gpu_frame_overrun(&self) -> f32 {
        branchless_clamp_f32(branchless_relu_f32(self.sliding_entropy_rate), 0.0, 1.0)
    }
    #[inline(always)]
    fn active_swarm_cells_mask(&self) -> u8 {
        let low_e = (self.active_branch_mode == 0)
            & (self.moving_z_entropy < 0.10)
            & (self.sliding_entropy_rate <= 0.5);
        branchless_select_u8(low_e, 0x0F, 0xFF)
    }
    #[inline(always)]
    fn recommend_layer_skip(&self) -> bool {
        (self.active_branch_mode == 0)
            & (self.moving_z_entropy <= 0.12)
            & (self.sliding_entropy_rate <= 0.5)
    }
}

impl CausalProvenanceSignal for RuntimeConfig {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_SYSTEM_CONSTRAINT
    }
}

impl SystemConstraintSignal for RuntimeConfig {
    #[inline(always)]
    fn system_vram_pressure(&self) -> f32 {
        0.0
    }
    #[inline(always)]
    fn phase_step_min_rad_override(&self) -> f32 {
        let valid =
            branchless_is_finite_f32(self.phase_step_min_rad) & (self.phase_step_min_rad >= 0.0);
        branchless_select_f32(valid, self.phase_step_min_rad, DEFAULT_PHASE_STEP_MIN_RAD)
    }
}

/// Cold-Path and startup/periodic environment-probe snapshot characterizing the active LLM's
/// RoPE subspace geometry, neutral entropy baseline, and inter-layer phase coherence.
///
/// Populated before startup and refreshed on periodic Cold-Path probes or model swaps so
/// [`StigmergicThresholdPolicy`] implementations can self-calibrate without hardcoded per-model constants.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ManifoldProbeBaseline {
    /// Active attention heads observed in the target model (`1..=MAX_ATTENTION_HEADS`).
    pub active_heads: u16,
    /// Number of transformer layers selected by `layer_rope_mask` (`>= 1`).
    pub steered_layers: u16,
    /// Baseline normalized `EntropyController` entropy observed on neutral calibration prompts ($\tilde{H}_0 \in [0, 1]$).
    pub baseline_entropy: f32,
    /// Mean positive inter-layer / intra-turn phase trajectory coherence measured during calibration ($\bar{c} \in [0, 1]$).
    pub baseline_inter_layer_coherence: f32,
    /// Neutral-token baseline cosine similarity against contrastive guard anchors ($\in [0, 1]$).
    pub baseline_guard_noise_floor: f32,
    /// Monotonically incremented Cold-Path calibration / re-allocation epoch counter.
    pub probe_epoch: u32,
}

impl Default for ManifoldProbeBaseline {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl ManifoldProbeBaseline {
    /// Creates a neutral default [`ManifoldProbeBaseline`] suitable before runtime probing runs.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            active_heads: MAX_ATTENTION_HEADS as u16,
            steered_layers: 16,
            baseline_entropy: 0.35,
            baseline_inter_layer_coherence: 0.75,
            baseline_guard_noise_floor: 0.45,
            probe_epoch: 0,
        }
    }
}

/// Trait abstracting dynamic derivation of [`StigmergicThresholds`] from hardware/telemetry
/// constraints (`SystemConstraintSignal`), Cold-Path model probe measurements (`ManifoldProbeBaseline`),
/// active pointer-trail state, and instantaneous entropy (`H_tilde`).
pub trait StigmergicThresholdPolicy {
    /// Resolves effective [`StigmergicThresholds`] for the current inference step without heap allocation.
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &PointerTrail,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds;

    /// Resolves [`StigmergicThresholds`] and specializes [`crate::stigmergy::StigmergicSubstrate`] decision gates using an
    /// optional [`CognitiveTrajectoryLedger`] habit state at `scale`.
    ///
    /// Blanket default delegates to [`Self::resolve_thresholds`] followed by
    /// [`CognitiveTrajectoryLedger::bias_stigmergic_thresholds`] so identical inputs trigger
    /// distinct regime ladders according to active cognitive circuit habits.
    #[inline(always)]
    fn resolve_thresholds_with_circuit<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &PointerTrail,
        entropy_h_tilde: f32,
        circuit: Option<&CognitiveTrajectoryLedger>,
        scale: StigmergicScaleGranularity,
    ) -> StigmergicThresholds {
        let base = self.resolve_thresholds(system, probe, trail, entropy_h_tilde);
        match circuit {
            Some(ledger) => ledger.bias_stigmergic_thresholds(base, scale),
            None => base,
        }
    }
}

impl<T: StigmergicThresholdPolicy + ?Sized> StigmergicThresholdPolicy for &T {
    #[inline(always)]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &PointerTrail,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        (**self).resolve_thresholds(system, probe, trail, entropy_h_tilde)
    }

    #[inline(always)]
    fn resolve_thresholds_with_circuit<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &PointerTrail,
        entropy_h_tilde: f32,
        circuit: Option<&CognitiveTrajectoryLedger>,
        scale: StigmergicScaleGranularity,
    ) -> StigmergicThresholds {
        (**self).resolve_thresholds_with_circuit(
            system,
            probe,
            trail,
            entropy_h_tilde,
            circuit,
            scale,
        )
    }
}

impl StigmergicThresholdPolicy for StigmergicThresholds {
    #[inline(always)]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        _system: &C,
        _probe: &ManifoldProbeBaseline,
        _trail: &PointerTrail,
        _entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        *self
    }
}

/// Closed-form, branchless adaptive threshold policy calibrating [`StigmergicThresholds`]
/// from live [`SystemConstraintSignal`] pressure, [`ManifoldProbeBaseline`] model geometry,
/// and pointer-trail stability.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveThresholdPolicy {
    /// Nominal baseline thresholds before runtime/probe calibration.
    pub base: StigmergicThresholds,
    /// Minimum/maximum clamp multiplier applied to dynamic gain & ladder adaptation.
    pub min_scale: f32,
    /// Maximum clamp multiplier applied to dynamic gain adaptation.
    pub max_scale: f32,
}

impl Default for AdaptiveThresholdPolicy {
    #[inline(always)]
    fn default() -> Self {
        Self::new(StigmergicThresholds::default())
    }
}

impl AdaptiveThresholdPolicy {
    /// Creates an [`AdaptiveThresholdPolicy`] anchored around `base`.
    #[inline(always)]
    #[must_use]
    pub const fn new(base: StigmergicThresholds) -> Self {
        Self {
            base,
            min_scale: 0.65,
            max_scale: 1.85,
        }
    }
}

/// Calibratable Q0.7 integer gating thresholds for [`CognitiveTrajectoryLedger`] cascades
/// and closed-loop self-reflection, avoiding hardcoded dataset-specific constants.
pub trait CognitiveCircuitThresholdPolicy {
    /// Minimum `fuse_gate_q7` required for adaptive early-exit (`0..=128`).
    fn early_exit_fuse_floor_q7(&self) -> u8 {
        96
    }

    /// Maximum `repulse_sensitivity_q7` allowed during early-exit (`0..=128`).
    fn early_exit_repulse_ceiling_q7(&self) -> u8 {
        72
    }

    /// Maximum mean residual phase deviation (`0..=128`) tolerated for early-exit.
    fn early_exit_residual_max_q7(&self) -> u8 {
        16
    }

    /// Minimum firing intensity (`0..=128`) triggering bounded 1-step recurrent refinement.
    fn recurrent_refine_intensity_floor_q7(&self) -> u8 {
        80
    }

    /// Minimum `peel_gate_q7` (`0..=128`) for escalating omitted compound decomposition.
    fn peel_escalation_floor_q7(&self) -> u8 {
        84
    }

    /// Derives the scale-invariant $S^{H-1}$ phase-resonance acceptance floor (`[24, 96]` in Q7)
    /// from the $2 / \sqrt{H}$ concentration-of-measure noise floor and cache occupancy pressure.
    #[inline]
    fn kv_resonance_floor_for_heads_q7(
        &self,
        active_heads: u16,
        occupied_slots: usize,
        capacity: usize,
    ) -> PhaseQ7 {
        let h = f32::from(active_heads.clamp(1, MAX_ATTENTION_HEADS as u16));
        let half = 0.5_f32 * h;
        let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(h.to_bits() >> 1));
        let y1 = y0 * (1.5_f32 - half * y0 * y0);
        let inv_sqrt_h = y1 * (1.5_f32 - half * y1 * y1);
        let base_floor = 2.0_f32 * inv_sqrt_h * 127.0_f32;
        let cap = capacity.max(1) as f32;
        let occ_ratio = branchless_clamp_f32(
            (occupied_slots as f32) * fast_inv_pos_normal_f32(cap),
            0.0,
            1.0,
        );
        let pressure_relax = 1.0_f32 - 0.15_f32 * occ_ratio;
        let raw = (base_floor * pressure_relax + 0.5_f32) as i32;
        raw.clamp(24, 96) as PhaseQ7
    }

    /// Returns the circular coherence threshold below which P-Frame seek steps trigger repulsion.
    #[inline(always)]
    fn kv_contradiction_threshold(&self) -> f32 {
        0.35
    }

    /// Derives the circuit composition blend weight (`16..=96` in Q7) from `frame`'s
    /// Fact-Persona coherence so aligned frames reinforce faster without hardcoded mixing ratios.
    #[inline(always)]
    fn kv_circuit_blend_q7(&self, frame: &GatewayFrame) -> u8 {
        let coh = if branchless_is_finite_f32(frame.fact_persona_coherence) {
            branchless_clamp_f32(frame.fact_persona_coherence, -1.0, 1.0)
        } else {
            0.0
        };
        let raw = ((0.5_f32 + 0.5_f32 * coh) * 96.0_f32 + 0.5_f32) as i32;
        raw.clamp(16, 96) as u8
    }
}

impl CognitiveCircuitThresholdPolicy for AdaptiveThresholdPolicy {
    #[inline(always)]
    fn early_exit_fuse_floor_q7(&self) -> u8 {
        let def_gain = crate::stigmergy::substrate::DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN;
        let coh_shift = ((self.base.reinforcement_gain.clamp(0.10, 1.50) - def_gain) * 32.0) as i32;
        (96 + coh_shift).clamp(72, 120) as u8
    }

    #[inline(always)]
    fn early_exit_residual_max_q7(&self) -> u8 {
        let scaled = ((16.0 * 1.85) / self.max_scale.clamp(0.8, 3.0)) as i32;
        scaled.clamp(8, 32) as u8
    }

    #[inline(always)]
    fn peel_escalation_floor_q7(&self) -> u8 {
        let def_evap = crate::stigmergy::substrate::DEFAULT_STIGMERGIC_EVAPORATION_RETAIN;
        let margin_adj =
            ((self.base.evaporation_retain.clamp(0.50, 0.99) - def_evap) * 80.0) as i32;
        (84 + margin_adj).clamp(64, 112) as u8
    }

    #[inline]
    fn kv_resonance_floor_for_heads_q7(
        &self,
        active_heads: u16,
        occupied_slots: usize,
        capacity: usize,
    ) -> PhaseQ7 {
        let h = f32::from(active_heads.clamp(1, MAX_ATTENTION_HEADS as u16));
        let half = 0.5_f32 * h;
        let y0 = f32::from_bits(0x5f37_5a86_u32.wrapping_sub(h.to_bits() >> 1));
        let y1 = y0 * (1.5_f32 - half * y0 * y0);
        let inv_sqrt_h = y1 * (1.5_f32 - half * y1 * y1);
        let def_gain = crate::stigmergy::substrate::DEFAULT_STIGMERGIC_REINFORCEMENT_GAIN;
        let gain_shift =
            ((self.base.reinforcement_gain.clamp(0.10, 1.50) - def_gain) * 16.0) as i32;
        let cap = capacity.max(1) as f32;
        let occ_ratio = branchless_clamp_f32(
            (occupied_slots as f32) * fast_inv_pos_normal_f32(cap),
            0.0,
            1.0,
        );
        let base_floor = 2.0_f32 * inv_sqrt_h * 127.0_f32 * (1.0_f32 - 0.15_f32 * occ_ratio);
        ((base_floor + 0.5_f32) as i32 + gain_shift).clamp(24, 96) as PhaseQ7
    }

    #[inline(always)]
    fn kv_contradiction_threshold(&self) -> f32 {
        branchless_clamp_f32(self.base.contradiction_coherence_thresh, 0.15, 0.75)
    }
}

impl StigmergicThresholdPolicy for AdaptiveThresholdPolicy {
    #[inline]
    fn resolve_thresholds<C: SystemConstraintSignal + ?Sized>(
        &self,
        system: &C,
        probe: &ManifoldProbeBaseline,
        trail: &PointerTrail,
        entropy_h_tilde: f32,
    ) -> StigmergicThresholds {
        let p_vram =
            branchless_clamp_f32(branchless_relu_f32(system.system_vram_pressure()), 0.0, 1.0);
        let p_overrun = branchless_clamp_f32(
            branchless_relu_f32(system.system_gpu_frame_overrun()),
            0.0,
            1.0,
        );
        let pressure = branchless_max_f32(p_vram, p_overrun);

        let base_coh = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(probe.baseline_inter_layer_coherence),
                probe.baseline_inter_layer_coherence,
                0.75,
            ),
            0.0,
            1.0,
        );
        let retain_0 = branchless_clamp_f32(
            branchless_relu_f32(self.base.evaporation_retain),
            0.0,
            0.995,
        );
        let is_subspace = probe.active_heads > 0 && probe.active_heads < MAX_ATTENTION_HEADS as u16;
        let eff_retain = if is_subspace {
            retain_0 * (0.85_f32 + 0.15_f32 * base_coh)
        } else {
            retain_0
        };
        let eff_traj = if is_subspace {
            0.38_f32 + 0.52_f32 * base_coh
        } else {
            0.50_f32 + 0.50_f32 * base_coh
        };
        // Equilibrate deposit gain against model-specific inter-layer decay & trajectory damping.
        // Reference equilibrium factor at default baseline (`c=0.75`, `retain=0.95`, `subspace`): 0.1444.
        let loss_over_traj2 = (1.0_f32 - eff_retain) * fast_inv_pos_normal_f32(eff_traj * eff_traj);
        let raw_gain_scale = loss_over_traj2 * (1.0_f32 / 0.1444_f32);
        let gain_scale = branchless_clamp_f32(raw_gain_scale, self.min_scale, self.max_scale);
        let pressure_damp = 1.0_f32 - 0.35_f32 * pressure;

        let dyn_gain = branchless_clamp_f32(
            self.base.reinforcement_gain * gain_scale * pressure_damp,
            0.05,
            1.25,
        );

        // Scale density ladder thresholds covariantly with the equilibrium density shift
        // (`dyn_gain / (1 - eff_retain)` relative to baseline) so low-coherence models do not
        // saturate into Rank-1 orchestration.
        let base_loss = (1.0_f32 - retain_0).max(0.01);
        let eff_loss = (1.0_f32 - eff_retain).max(0.01);
        let base_eq = self.base.reinforcement_gain * fast_inv_pos_normal_f32(base_loss);
        let dyn_eq = dyn_gain * fast_inv_pos_normal_f32(eff_loss);
        let eq_scale = branchless_clamp_f32(
            dyn_eq * fast_inv_pos_normal_f32(base_eq.max(0.05)),
            0.85,
            2.00,
        );

        // Sustained high-pull trails gradually ease crystallization, while resource pressure raises skip/Rank-1 residual gates.
        let sustained_pull = if trail.step_count >= 16 {
            branchless_clamp_f32((trail.attractor_pull_force - 0.65_f32) * 2.0_f32, 0.0, 1.0)
        } else {
            0.0_f32
        };
        let ref_h0 = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(probe.baseline_entropy) & (probe.baseline_entropy > 0.01),
                probe.baseline_entropy,
                0.35,
            ),
            0.10,
            0.90,
        );
        let h0_scale = branchless_clamp_f32(ref_h0 * (1.0_f32 / 0.35_f32), 0.80, 1.45);
        let skip_thresh = branchless_clamp_f32(
            self.base.inertial_skip_entropy_thresh * h0_scale * (1.0_f32 + 0.75_f32 * pressure),
            0.05,
            0.45,
        );
        let express_thresh =
            self.base.express_steer_density_thresh * eq_scale * (1.0_f32 + 0.35_f32 * pressure);
        let subhead_thresh =
            self.base.subhead_route_density_thresh * eq_scale * (1.0_f32 + 0.30_f32 * pressure);
        let rank1_thresh =
            self.base.rank1_inject_density_thresh * eq_scale * (1.0_f32 + 0.50_f32 * pressure);
        let anchor_thresh = self.base.crystallize_density_thresh
            * eq_scale
            * (1.0_f32 - 0.15_f32 * sustained_pull + 0.25_f32 * pressure);

        // Gently relax contradiction sensitivity when probe observes noisy guard floor or high entropy.
        let h_valid = branchless_select_f32(
            branchless_is_finite_f32(entropy_h_tilde),
            branchless_clamp_f32(entropy_h_tilde, 0.0, 1.0),
            0.35,
        );
        let noise_adj = branchless_clamp_f32(
            branchless_relu_f32(probe.baseline_guard_noise_floor - 0.50_f32) * 0.40_f32,
            0.0,
            0.25,
        );
        let contradict_thresh = branchless_clamp_f32(
            self.base.contradiction_coherence_thresh + noise_adj + 0.05_f32 * h_valid,
            0.15,
            0.75,
        );

        StigmergicThresholds {
            evaporation_retain: retain_0,
            reinforcement_gain: dyn_gain,
            inertial_skip_entropy_thresh: skip_thresh,
            express_steer_density_thresh: express_thresh,
            subhead_route_density_thresh: subhead_thresh,
            rank1_inject_density_thresh: rank1_thresh,
            crystallize_density_thresh: anchor_thresh,
            contradiction_coherence_thresh: contradict_thresh,
        }
    }
}

// ===================================================================
// 5. Origin 2: Cognitive Grounding Hierarchy (`Fact` x `Persona` x `ContradictionGuard`)
// ===================================================================

/// **Origin 2 Parent Supertrait (`CognitiveGroundingSignal: CausalProvenanceSignal`)**:
/// Common phase-projection and `.lmnk` (mapped-storage) 48-bit section-offset addressing foundation.
pub trait CognitiveGroundingSignal: CausalProvenanceSignal {
    /// Static cognitive aperture spanning Origin 2A (Fact), 2B (Persona), and 2C (Guard) (`0x0E`).
    const GROUNDING_APERTURE_MASK: u32 =
        THRESHOLD_OBJECTIVE_FACT | THRESHOLD_SUBJECTIVE_PERSONA | THRESHOLD_CONTRADICTION_GUARD;

    /// Complement bitmask outside Origin 2 semantic grounding (`!GROUNDING_APERTURE_MASK`).
    const GROUNDING_UNOBSERVED_LACUNA_MASK: u32 = !Self::GROUNDING_APERTURE_MASK;

    /// Projects this grounding signal onto the 64-head $SO(2)^{64}$ Q7 phase manifold.
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS];

    /// Fast zero-copy borrow of the 64-head Q7 phase profile when already materialized in memory.
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[PhaseQ7; MAX_ATTENTION_HEADS]> {
        None
    }

    /// Projects this grounding signal directly into 8 register-resident `u64` SWAR words (`[u64; 8]`).
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        match self.grounding_phases_ref_64() {
            Some(arr_ref) => load_q7_words_64(arr_ref),
            None => load_q7_words_64(&self.project_grounding_phases_64()),
        }
    }

    /// Base gravitational pull energy $E_{\mathrm{base}} \ge 0$ of this grounding signal.
    fn grounding_base_energy(&self) -> f32;

    /// Canonical 48-bit `.lmnk` (mapped-storage) section or RAG record offset (`bits 0..=47` of `TaggedPtr`).
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        0
    }

    /// Full 32-bit macro attractor anchor identifier on the Single Unified Attractor Manifold.
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        0
    }

    /// Active attention-head subspace dimension $H \in [1, 64]$ over which phase coherence is evaluated.
    #[inline(always)]
    fn active_head_count(&self) -> u16 {
        MAX_ATTENTION_HEADS as u16
    }

    /// Single-pass grounding envelope `(phase_words, base_energy, mapped_storage_offset48, anchor_id, active_heads)`
    /// avoiding redundant slice folds on composite or resonant signals.
    #[inline(always)]
    fn synthesize_grounding_envelope(
        &self,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
        (
            self.project_grounding_words_64(),
            self.grounding_base_energy(),
            self.mapped_section_offset48(),
            self.grounding_anchor_id(),
            self.active_head_count(),
        )
    }
}

impl<T: CognitiveGroundingSignal + ?Sized> CognitiveGroundingSignal for &T {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        (**self).project_grounding_phases_64()
    }
    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[PhaseQ7; MAX_ATTENTION_HEADS]> {
        (**self).grounding_phases_ref_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        (**self).project_grounding_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        (**self).grounding_base_energy()
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        (**self).mapped_section_offset48()
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        (**self).grounding_anchor_id()
    }
    #[inline(always)]
    fn active_head_count(&self) -> u16 {
        (**self).active_head_count()
    }
    #[inline(always)]
    fn synthesize_grounding_envelope(
        &self,
    ) -> ([u64; NUM_STIGMERGIC_GRANULES], f32, u64, u32, u16) {
        (**self).synthesize_grounding_envelope()
    }
}

/// **Origin 2-A Subtrait (`ObjectiveFactSignal: CognitiveGroundingSignal`)**:
/// External RAG chunks, database records, and objective conversation logs (`Slot 2`, narrow `PhaseClamp` limit `0.14 rad`).
pub trait ObjectiveFactSignal: CognitiveGroundingSignal {
    /// Narrow single-axis cognitive aperture for factual grounding (`1 << 1`).
    const FACT_APERTURE_MASK: u32 = THRESHOLD_OBJECTIVE_FACT;

    /// Complement bitmask treating non-factual axes as unknowable (`!THRESHOLD_OBJECTIVE_FACT`).
    const FACT_UNOBSERVED_LACUNA_MASK: u32 = !Self::FACT_APERTURE_MASK;

    /// `PhaseClamp` limit in radians for objective factual grounding (defaults to `0.14 rad`).
    #[inline(always)]
    fn fact_phase_clamp_rad(&self) -> f32 {
        DEFAULT_FACT_PHASE_CLAMP_RAD
    }

    /// sub-head router `Fact` sub-head phase gain multiplier (defaults to `1.15`).
    #[inline(always)]
    fn fact_subhead_gain(&self) -> f32 {
        1.15
    }
}

impl<T: ObjectiveFactSignal + ?Sized> ObjectiveFactSignal for &T {
    #[inline(always)]
    fn fact_phase_clamp_rad(&self) -> f32 {
        (**self).fact_phase_clamp_rad()
    }
    #[inline(always)]
    fn fact_subhead_gain(&self) -> f32 {
        (**self).fact_subhead_gain()
    }
}

/// **Origin 2-B Subtrait (`SubjectivePersonaSignal: CognitiveGroundingSignal`)**:
/// System prompts, character beliefs, stylistic tone, and subjective episodic memory (`Slot 1`, wide `PhaseClamp` limit `0.24 rad`).
pub trait SubjectivePersonaSignal: CognitiveGroundingSignal {
    /// Narrow single-axis cognitive aperture for subjective persona tone (`1 << 2`).
    const PERSONA_APERTURE_MASK: u32 = THRESHOLD_SUBJECTIVE_PERSONA;

    /// Complement bitmask treating non-persona axes as unknowable (`!THRESHOLD_SUBJECTIVE_PERSONA`).
    const PERSONA_UNOBSERVED_LACUNA_MASK: u32 = !Self::PERSONA_APERTURE_MASK;

    /// `PhaseClamp` limit in radians for subjective persona modulation (defaults to `0.24 rad`).
    #[inline(always)]
    fn persona_phase_clamp_rad(&self) -> f32 {
        DEFAULT_PERSONA_PHASE_CLAMP_RAD
    }

    /// sub-head router `Explore` / `Logic` sub-head phase gain multiplier (defaults to `1.20`).
    #[inline(always)]
    fn persona_subhead_gain(&self) -> f32 {
        1.20
    }
}

impl<T: SubjectivePersonaSignal + ?Sized> SubjectivePersonaSignal for &T {
    #[inline(always)]
    fn persona_phase_clamp_rad(&self) -> f32 {
        (**self).persona_phase_clamp_rad()
    }
    #[inline(always)]
    fn persona_subhead_gain(&self) -> f32 {
        (**self).persona_subhead_gain()
    }
}

/// **Origin 2-C Subtrait (`ContradictionGuardSignal: CognitiveGroundingSignal`)**:
/// Sheaf cohomology obstruction ($H^1(K; \mathcal{F}) \neq 0$) and [`ContrastivePhaseBloomFilter`]
/// repetition / self-contradiction guard triggering anti-phase $-\boldsymbol{\Delta\theta}$ escape.
pub trait ContradictionGuardSignal: CognitiveGroundingSignal {
    /// Single-axis cognitive aperture for contradiction guard repulsion (`1 << 3`).
    const GUARD_APERTURE_MASK: u32 = THRESHOLD_CONTRADICTION_GUARD;

    /// Complement bitmask treating non-guard axes as unknowable (`!THRESHOLD_CONTRADICTION_GUARD`).
    const GUARD_UNOBSERVED_LACUNA_MASK: u32 = !Self::GUARD_APERTURE_MASK;

    /// Evaluates repulsive severity in $[0.0, 1.0]$ against `candidate_phases`.
    fn evaluate_repulsion_severity(&self, candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS])
    -> f32;

    /// Evaluates repulsive severity directly from 8 register-resident `u64` SWAR words (`candidate_words`).
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let _ = candidate_words;
        self.evaluate_repulsion_severity(candidate_phases)
    }

    /// Returns `true` if [`Self::evaluate_repulsion_severity_from_words`] inspects `candidate_phases`
    /// rather than operating purely on `candidate_words` in registers.
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        true
    }

    /// Projects the repulsive pole's positive phase words on $SO(2)^{64}$, or all zeros when this
    /// guard operates purely as a phase-less candidate detector (such as a Bloom filter or scalar
    /// obstruction guard).
    #[inline(always)]
    fn project_repulsive_pole_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.project_grounding_words_64()
    }
}

impl<T: ContradictionGuardSignal + ?Sized> ContradictionGuardSignal for &T {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        (**self).evaluate_repulsion_severity(candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        (**self).evaluate_repulsion_severity_from_words(candidate_words, candidate_phases)
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        (**self).requires_phase_slice_for_repulsion()
    }

    #[inline(always)]
    fn project_repulsive_pole_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        (**self).project_repulsive_pole_words_64()
    }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as an **Objective Fact** (`Origin 2-A`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectiveFact<T> {
    /// Underlying raw payload (`&str`, `&[u8]`, `&[f32]`, or `[PhaseQ7; 64]`).
    pub payload: T,
    /// Base gravitational pull energy $E_{\mathrm{fact}} \ge 0$.
    pub base_energy: f32,
    /// Direct 48-bit `.lmnk` (mapped-storage) section offset (`bits 0..=47`).
    pub mapped_storage_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// `PhaseClamp` limit in radians (defaults to [`DEFAULT_FACT_PHASE_CLAMP_RAD`]).
    pub phase_clamp_rad: f32,
}

impl<T> ObjectiveFact<T> {
    /// Constructs a new [`ObjectiveFact`] with explicit base energy, 48-bit mapped-storage offset, and 32-bit anchor ID.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        payload: T,
        base_energy: f32,
        mapped_storage_offset48: u64,
        anchor_id: u32,
    ) -> Self {
        Self {
            payload,
            base_energy: branchless_flush_subnormal_f32(branchless_relu_f32(base_energy)),
            mapped_storage_offset48: mapped_storage_offset48 & TaggedPtr::ADDR_MASK,
            anchor_id,
            phase_clamp_rad: DEFAULT_FACT_PHASE_CLAMP_RAD,
        }
    }

    /// Builder method overriding the `PhaseClamp` limit in radians.
    #[inline(always)]
    #[must_use]
    pub const fn with_phase_clamp_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.phase_clamp_rad = branchless_select_f32(valid, max_rad, self.phase_clamp_rad);
        self
    }
}

impl<T: RawPhaseProjectable> CausalProvenanceSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let weak = self.base_energy < 0.25;
        CAUSAL_AXIS_OBJECTIVE_FACT
            | branchless_select_u32(weak, CAUSAL_CAUSE_FACT_RETRIEVAL_WEAK, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_select_f32(self.base_energy < 0.25, 0.35, 0.0)
    }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.payload.project_to_q7_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.mapped_storage_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl<T: RawPhaseProjectable> ObjectiveFactSignal for ObjectiveFact<T> {
    #[inline(always)]
    fn fact_phase_clamp_rad(&self) -> f32 {
        self.phase_clamp_rad
    }
}

/// Concrete wrapper classifying any [`RawPhaseProjectable`] payload `T` as a **Subjective Persona** (`Origin 2-B`).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubjectivePersona<T> {
    /// Underlying raw payload (`&str`, `&[u8]`, `&[f32]`, or `[PhaseQ7; 64]`).
    pub payload: T,
    /// Base gravitational pull energy $E_{\mathrm{persona}} \ge 0$.
    pub base_energy: f32,
    /// Direct 48-bit `.lmnk` (mapped-storage) persona section offset (`bits 0..=47`).
    pub mapped_storage_offset48: u64,
    /// Full 32-bit macro anchor ID on the unified manifold.
    pub anchor_id: u32,
    /// `PhaseClamp` limit in radians (defaults to [`DEFAULT_PERSONA_PHASE_CLAMP_RAD`]).
    pub phase_clamp_rad: f32,
}

impl<T> SubjectivePersona<T> {
    /// Constructs a new [`SubjectivePersona`] with explicit base energy, 48-bit mapped-storage offset, and 32-bit anchor ID.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        payload: T,
        base_energy: f32,
        mapped_storage_offset48: u64,
        anchor_id: u32,
    ) -> Self {
        Self {
            payload,
            base_energy: branchless_flush_subnormal_f32(branchless_relu_f32(base_energy)),
            mapped_storage_offset48: mapped_storage_offset48 & TaggedPtr::ADDR_MASK,
            anchor_id,
            phase_clamp_rad: DEFAULT_PERSONA_PHASE_CLAMP_RAD,
        }
    }

    /// Builder method overriding the `PhaseClamp` limit in radians.
    #[inline(always)]
    #[must_use]
    pub const fn with_phase_clamp_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.phase_clamp_rad = branchless_select_f32(valid, max_rad, self.phase_clamp_rad);
        self
    }
}

impl<T: RawPhaseProjectable> CausalProvenanceSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_SUBJECTIVE_PERSONA
    }
}

impl<T: RawPhaseProjectable> CognitiveGroundingSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.payload.project_to_q7_64()
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.payload.project_to_words_64()
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.base_energy
    }
    #[inline(always)]
    fn mapped_section_offset48(&self) -> u64 {
        self.mapped_storage_offset48
    }
    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.anchor_id
    }
}

impl<T: RawPhaseProjectable> SubjectivePersonaSignal for SubjectivePersona<T> {
    #[inline(always)]
    fn persona_phase_clamp_rad(&self) -> f32 {
        self.phase_clamp_rad
    }
}

/// Concrete wrapper for **Origin 2-C (`ContradictionGuardSignal`)** combining an optional
/// [`ContrastivePhaseBloomFilter`] (or `()`) with a sheaf cohomology obstruction residual.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContradictionGuard<B = ContrastivePhaseBloomFilter> {
    /// Underlying Bloom filter or unit `()`.
    pub bloom: B,
    /// Observed cohomological obstruction residual norm $\| \mathbf{r}_{\mathrm{obs}} \| \ge 0$.
    pub cohomology_obstruction: f32,
    /// Threshold above which `cohomology_obstruction` triggers repulsive escape (defaults to `0.25`).
    pub obstruction_threshold: f32,
    /// Repulsive Bloom bucket density threshold (`1..=15`, defaults to `4`).
    pub bloom_repulsion_threshold: u8,
}

impl<B> ContradictionGuard<B> {
    /// Constructs a new [`ContradictionGuard`] with explicit cohomological obstruction and Bloom threshold.
    #[inline(always)]
    #[must_use]
    pub const fn new(bloom: B, cohomology_obstruction: f32, bloom_repulsion_threshold: u8) -> Self {
        Self {
            bloom,
            cohomology_obstruction: branchless_flush_subnormal_f32(branchless_relu_f32(
                cohomology_obstruction,
            )),
            obstruction_threshold: 0.25,
            bloom_repulsion_threshold: branchless_select_u8(
                bloom_repulsion_threshold > 0,
                bloom_repulsion_threshold,
                4,
            ),
        }
    }

    /// Builder method setting the cohomological obstruction threshold.
    #[inline(always)]
    #[must_use]
    pub const fn with_obstruction_threshold(mut self, thresh: f32) -> Self {
        let valid = branchless_is_finite_f32(thresh) & (thresh > 0.0);
        self.obstruction_threshold =
            branchless_select_f32(valid, thresh, self.obstruction_threshold);
        self
    }
}

impl<B> CausalProvenanceSignal for ContradictionGuard<B> {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD
            | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }
    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        if self.cohomology_obstruction <= 0.0 {
            return 0.0;
        }
        let inv_thresh = if self.obstruction_threshold == 0.25 {
            4.0
        } else if self.obstruction_threshold >= 1.0e-30 {
            fast_inv_pos_normal_f32(self.obstruction_threshold)
        } else {
            branchless_inv_pos_f32(self.obstruction_threshold)
        };
        let ratio = self.cohomology_obstruction * inv_thresh;
        branchless_clamp_f32(ratio * 0.5, 0.0, 1.0)
    }
}

impl<B> CognitiveGroundingSignal for ContradictionGuard<B> {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.cohomology_obstruction
    }
}

impl ContradictionGuardSignal for ContradictionGuard<ContrastivePhaseBloomFilter> {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        let obs_sev = branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        );
        if obs_sev >= 1.0 {
            return 1.0;
        }
        let density = self
            .bloom
            .query_repulsive_density_from_words(candidate_words);
        let bloom_hit = density >= self.bloom_repulsion_threshold;
        let bloom_sev = branchless_select_f32(
            bloom_hit,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        );
        branchless_max_f32(obs_sev, bloom_sev)
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

impl ContradictionGuardSignal for ContradictionGuard<()> {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        _candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

/// 64-byte Q7 phase anchor for contrastive/negative guard attractors, enabling
/// query-resonance repulsion checks inside [`ResonantSignalSlice`] and 3-origin tuples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardPhaseAnchor {
    /// Per-head Q7 phase deltas (`[i8; 64]`).
    pub phase_deltas: [PhaseQ7; MAX_ATTENTION_HEADS],
    /// Observed cohomological obstruction residual norm $\| \mathbf{r}_{\mathrm{obs}} \| \ge 0$.
    pub cohomology_obstruction: f32,
    /// Threshold above which `cohomology_obstruction` triggers repulsive escape.
    pub obstruction_threshold: f32,
}

impl GuardPhaseAnchor {
    /// Constructs a new [`GuardPhaseAnchor`] from a 64-head Q7 phase array.
    #[inline(always)]
    #[must_use]
    pub const fn new(
        phase_deltas: [PhaseQ7; MAX_ATTENTION_HEADS],
        cohomology_obstruction: f32,
        obstruction_threshold: f32,
    ) -> Self {
        Self {
            phase_deltas,
            cohomology_obstruction: branchless_flush_subnormal_f32(branchless_relu_f32(
                cohomology_obstruction,
            )),
            obstruction_threshold,
        }
    }
}

impl CausalProvenanceSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        CAUSAL_AXIS_CONTRADICTION_GUARD
            | branchless_select_u32(obstructed, CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        let obstructed = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obstructed,
            branchless_clamp_f32(self.cohomology_obstruction, 0.0, 1.0),
            0.0,
        )
    }
}

impl CognitiveGroundingSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.phase_deltas.project_to_words_64()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.cohomology_obstruction
    }
}

impl ContradictionGuardSignal for GuardPhaseAnchor {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        _candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let obs_exceeded = self.cohomology_obstruction > self.obstruction_threshold;
        branchless_select_f32(
            obs_exceeded,
            branchless_clamp_f32(
                0.55 + (self.cohomology_obstruction - self.obstruction_threshold),
                0.55,
                1.0,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}

impl CausalProvenanceSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        CAUSAL_AXIS_CONTRADICTION_GUARD
    }
}

impl CognitiveGroundingSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        [0i8; MAX_ATTENTION_HEADS]
    }
    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        [0u64; NUM_STIGMERGIC_GRANULES]
    }
    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        0.0
    }
}

impl ContradictionGuardSignal for ContrastivePhaseBloomFilter {
    #[inline(always)]
    fn evaluate_repulsion_severity(
        &self,
        candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let words = load_q7_words_64(candidate_phases);
        self.evaluate_repulsion_severity_from_words(&words, candidate_phases)
    }
    #[inline(always)]
    fn evaluate_repulsion_severity_from_words(
        &self,
        candidate_words: &[u64; NUM_STIGMERGIC_GRANULES],
        _candidate_phases: &[PhaseQ7; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let density = self.query_repulsive_density_from_words(candidate_words);
        branchless_select_f32(
            density >= 3,
            branchless_clamp_f32(0.60 + (density as f32) * 0.05, 0.60, 1.0),
            0.0,
        )
    }
    #[inline(always)]
    fn requires_phase_slice_for_repulsion(&self) -> bool {
        false
    }
}
