// name: src/core/src/drivers/distortion.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single shared interface between `drivers/spine` and `drivers/ego`:
//! directed P-Frame [`DistortionDelta`] packets (128 B), asymmetric homeostatic
//! [`DistortionTension`] (32 B), and [`LayerDistortionKind`] / [`LayerDistortionReport`] value types.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::drivers::ego::EgoIntentAxes;
use crate::foundation::Pod;
use crate::liminal::{
    branchless_select_u8, gcso_q7_t, AttributionTag, MAX_ATTENTION_HEADS,
    SCOPE_REGION_SELF_EXCLUSIVE, SCOPE_REGION_SHARED_CORE,
};

/// Multiplies signed Q7 phase `delta` by unsigned Q0.7 `scale_u7` (`0..=128`) branchlessly.
#[inline(always)]
#[must_use]
#[allow(dead_code)]
pub(crate) const fn scale_q7_by_u7(delta: gcso_q7_t, scale_u7: u8) -> gcso_q7_t {
    let clamped = if scale_u7 > 128 { 128 } else { scale_u7 };
    let prod = (delta as i32) * (clamped as i32);
    let out = prod >> 7;
    if out > 127 {
        127
    } else if out < -127 {
        -127
    } else {
        out as gcso_q7_t
    }
}

/// Blends signed Q7 value `cur` toward `target` by weight `alpha_u7` (`0..=128`) branchlessly.
#[inline(always)]
#[must_use]
#[allow(dead_code)]
pub(crate) const fn blend_q7_toward(cur: gcso_q7_t, target: gcso_q7_t, alpha_u7: u8) -> gcso_q7_t {
    let w = if alpha_u7 > 128 { 128 } else { alpha_u7 } as i32;
    let inv = 128 - w;
    let out = ((cur as i32) * inv + (target as i32) * w) >> 7;
    if out > 127 {
        127
    } else if out < -127 {
        -127
    } else {
        out as gcso_q7_t
    }
}

/// Maximum number of inline P-Frame [`DistortionDelta`] nodes hosted by `EgoDifferentialNet`.
pub const MAX_EGO_PFRAME_NODES: usize = 8;

/// Sentinel next-hop index marking termination of a [`DistortionDelta`] pointer chain.
pub const DISTORTION_DELTA_NO_NEXT_HOP: u16 = 0xFFFF;

/// Default maximum hop depth allowed when evaluating branched [`DistortionDelta`] chains (`4`).
pub const MAX_DISTORTION_CHAIN_HOPS: u8 = 4;

/// Flag bit `0`: [`DistortionDelta`] node is active and eligible for Venn trigger matching.
pub const DISTORTION_DELTA_FLAG_ACTIVE: u32 = 1 << 0;

/// Flag bit `1`: [`DistortionDelta`] node was consolidated into an I-Frame crystal anchor.
pub const DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME: u32 = 1 << 1;

/// Flag bit `2`: [`DistortionDelta`] node carries valid model-baked 64-head Q7 phase deltas.
pub const DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID: u32 = 1 << 2;

// ===================================================================
// 4. Directed P-Frame Packet (`DistortionDelta`, 128 B)
// ===================================================================

/// First-class 128-byte (`#[repr(C, align(64))]`) directed P-Frame differential node representing
/// a gauge-relative deviation $\vec{\Delta}_t$ from `SystemSpineState` trajectory $f_{\mathrm{sys}}$.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistortionDelta {
    /// Logical P-Frame node identifier (`offset 0..4`).
    pub node_id: u32,
    /// Reference logical I-Frame anchor identifier (`offset 4..8`).
    pub anchor_ref_id: u32,
    /// Next chained P-Frame node index (`0..8`, or `0xFFFF` if terminal, `offset 8..10`).
    pub next_hop_idx: u16,
    /// Hop depth along the chain (`1..=4`, `offset 10`).
    pub hop_depth: u8,
    /// 8-granule Venn support bitmask required to trigger this delta (`offset 11`).
    pub venn_trigger_mask: u8,
    /// Bitmask of allowed Venn regions (`bit 0=S&O`, `bit 1=S\O`, `bit 2=O\S`, `offset 12`).
    pub venn_region_selector: u8,
    /// Stigmergic trace density in Q0.7 (`0..=127`, `offset 13`).
    pub pheromone_q7: u8,
    /// Directed delta scaling gain in Q0.7 (`0..=127`, `offset 14`).
    pub gain_q7: u8,
    /// 2-bit [`AttributionTag`] (`Stigmergy`/`Ego`, `offset 15`).
    pub attribution_tag: u8,
    /// Model-agnostic logical `EgoIntentAxes` transition (`16 B`, `offset 16..32`).
    pub intent_delta: EgoIntentAxes,
    /// Accumulated positive reinforcement counter for Cold-Path I-Frame consolidation (`offset 32..36`).
    pub reinforce_count: u32,
    /// Last active step epoch for stigmergic evaporation pruning (`offset 36..40`).
    pub last_active_epoch: u32,
    /// Optional integrity checksum over logical delta payload (`offset 40..44`).
    pub checksum_crc32: u32,
    /// Lifecycle and validity bitflags (`DISTORTION_DELTA_FLAG_*`, `offset 44..48`).
    pub flags: u32,
    /// Reserved aligned header bytes (`offset 48..64`).
    pub reserved_header: [u8; 16],
    /// Model-baked 64-head gauge-relative phase delta $\vec{\Delta}_t$ (`offset 64..128`).
    pub delta_phase_q7: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

impl Default for DistortionDelta {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(0, 0)
    }
}

impl DistortionDelta {
    /// Constructs a zeroed inactive 128-byte [`DistortionDelta`] packet.
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(node_id: u32, anchor_ref_id: u32) -> Self {
        Self {
            node_id,
            anchor_ref_id,
            next_hop_idx: DISTORTION_DELTA_NO_NEXT_HOP,
            hop_depth: 1,
            venn_trigger_mask: 0xFF,
            venn_region_selector: SCOPE_REGION_SHARED_CORE | SCOPE_REGION_SELF_EXCLUSIVE,
            pheromone_q7: 64,
            gain_q7: 64,
            attribution_tag: AttributionTag::STIGMERGY_EGO.raw(),
            intent_delta: EgoIntentAxes::new(),
            reinforce_count: 0,
            last_active_epoch: 0,
            checksum_crc32: 0,
            flags: 0,
            reserved_header: [0; 16],
            delta_phase_q7: [0; MAX_ATTENTION_HEADS],
        }
    }

    /// Returns `true` when this node is marked active (`DISTORTION_DELTA_FLAG_ACTIVE`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.flags & DISTORTION_DELTA_FLAG_ACTIVE) != 0
    }

    /// Returns `true` when this node is active (`DISTORTION_DELTA_FLAG_ACTIVE`) and carries
    /// valid model-baked 64-head deltas (`DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID`).
    #[inline(always)]
    #[must_use]
    pub const fn is_baked_active(&self) -> bool {
        const MASK: u32 = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        (self.flags & MASK) == MASK
    }
}

// ===================================================================
// 5. Asymmetric Homeostatic Tension (`DistortionTension`, 32 B)
// ===================================================================

/// Classification states of [`DistortionTension`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistortionTensionStateCode {
    /// Healthy differential torsion inside Deadband window `[deadband_low_q7, deadband_high_q7]` (`0`).
    #[default]
    BalancedDeadband = 0,
    /// Slow-Loop accumulation exciting Ego out of passive stagnation (`1`).
    PassiveExcitation = 1,
    /// Immediate 1-step Hot-Path damping on ActiveOverdrive / entropy spike / guard proximity (`2`).
    ActiveImmediateDamping = 2,
}

/// Asymmetric homeostatic field tension (`32 B`, `#[repr(C, align(16))]`) mediating the tug-of-war
/// between `SystemSpineState` convergence and `EgoDifferentialNet` deviation.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DistortionTension {
    /// Instantaneous tension magnitude in Q0.7 (`0..=127`, `offset 0`).
    pub tension_q7: u8,
    /// Lower Deadband torsion floor in Q0.7 below which stagnation builds (`offset 1`).
    pub deadband_low_q7: u8,
    /// Upper Deadband torsion ceiling in Q0.7 above which immediate damping fires (`offset 2`).
    pub deadband_high_q7: u8,
    /// Effective multiplicative Q0.7 scale applied to [`DistortionDelta`] amplitudes (`offset 3`).
    pub effective_ego_scale_q7: u8,
    /// Slow-Loop accumulator (`0..=255`) driving gradual Ego awakening (`offset 4`).
    pub excitation_potential_u8: u8,
    /// Remaining Hot-Path cooldown steps after immediate damping (`offset 5`).
    pub suppression_cooldown_steps: u8,
    /// Latest $4p(1-p)$ boundary sensitivity in Q0.7 (`offset 6`).
    pub boundary_sensitivity_q7: u8,
    /// Active [`DistortionTensionStateCode`] byte (`offset 7`).
    pub state_code: u8,
    /// Cumulative count of 1-step Hot-Path immediate damping interventions (`offset 8..12`).
    pub immediate_damping_events: u32,
    /// Cumulative count of Slow-Loop stagnation excitations (`offset 12..16`).
    pub slow_excitation_events: u32,
    /// L-infinity / mean magnitude of the latest differential $\delta_{\mathrm{ego}}$ (`offset 16`).
    pub last_ego_delta_norm_q7: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`, `Ego`, `offset 17`).
    pub attribution_tag: u8,
    /// Observed non-pinned guard proximity in Q0.7 (`offset 18`).
    pub guard_proximity_q7: u8,
    /// Observed exit entropy in Q0.7 (`offset 19`).
    pub exit_entropy_q7: u8,
    /// Cumulative L1 deviation norm tracked across steps (`offset 20..24`).
    pub cumulative_delta_l1: u32,
    /// Monotonic Hot-Path evaluation count (`offset 24..28`).
    pub step_epochs: u32,
    /// Explicit C alignment padding (`offset 28..32`).
    pub reserved_align: [u8; 4],
}

impl Default for DistortionTension {
    #[inline(always)]
    fn default() -> Self {
        Self::new(false)
    }
}

impl DistortionTension {
    /// Constructs a neutral [`DistortionTension`] controller (`effective_ego_scale_q7 = 0` when `!ego_enabled`,
    /// or `96` with Deadband `[16, 84]` when `ego_enabled`).
    #[inline(always)]
    #[must_use]
    pub const fn new(ego_enabled: bool) -> Self {
        Self {
            tension_q7: 0,
            deadband_low_q7: 16,
            deadband_high_q7: 84,
            effective_ego_scale_q7: if ego_enabled { 96 } else { 0 },
            excitation_potential_u8: 0,
            suppression_cooldown_steps: 0,
            boundary_sensitivity_q7: 0,
            state_code: DistortionTensionStateCode::BalancedDeadband as u8,
            immediate_damping_events: 0,
            slow_excitation_events: 0,
            last_ego_delta_norm_q7: 0,
            attribution_tag: AttributionTag::LIMINAL_EGO.raw(),
            guard_proximity_q7: 0,
            exit_entropy_q7: 0,
            cumulative_delta_l1: 0,
            step_epochs: 0,
            reserved_align: [0; 4],
        }
    }

    /// Executes 1 Hot-Path tension step (`0 alloc`, `0 fdiv`, `0` data-dependent branches).
    ///
    /// Asymmetric fast safety response: when `active_overdrive`, `exit_entropy_q7 >= 84`,
    /// `guard_proximity_q7 >= 72`, or `tension_q7 > deadband_high_q7` occurs, suppresses
    /// `effective_ego_scale_q7` in 1 step and arms `suppression_cooldown_steps = 3`.
    #[inline]
    pub fn step_hot_path_branchless(
        &mut self,
        ego_delta_norm_q7: u8,
        boundary_sensitivity_q7: u8,
        exit_entropy_q7: u8,
        guard_proximity_q7: u8,
        active_overdrive: bool,
    ) {
        self.last_ego_delta_norm_q7 = ego_delta_norm_q7.min(127);
        self.boundary_sensitivity_q7 = boundary_sensitivity_q7.min(127);
        self.exit_entropy_q7 = exit_entropy_q7.min(127);
        self.guard_proximity_q7 = guard_proximity_q7.min(127);
        self.cumulative_delta_l1 = self
            .cumulative_delta_l1
            .saturating_add(u32::from(self.last_ego_delta_norm_q7));
        self.step_epochs = self.step_epochs.saturating_add(1);

        // Compute instantaneous torsion mitigated inside the Deadband at high boundary sensitivity.
        let sens_relief = boundary_sensitivity_q7 >> 3;
        let raw_tension = self
            .last_ego_delta_norm_q7
            .saturating_add(self.guard_proximity_q7 >> 1)
            .saturating_add(self.exit_entropy_q7 >> 2)
            .saturating_sub(sens_relief)
            .min(127);
        self.tension_q7 = raw_tension;

        let spike = active_overdrive
            | (self.exit_entropy_q7 >= 84)
            | (self.guard_proximity_q7 >= 72)
            | (raw_tension > self.deadband_high_q7);
        let severe_guard = self.guard_proximity_q7 >= 96;

        let quarter = self.effective_ego_scale_q7 >> 2;
        let mild_floor = branchless_select_u8(self.effective_ego_scale_q7 > 0, quarter.max(8), 0);
        let damped_scale = branchless_select_u8(severe_guard, 0, mild_floor);
        let cooled = self.suppression_cooldown_steps.saturating_sub(1);
        let healthy_recover = branchless_select_u8(
            (cooled == 0) & (self.effective_ego_scale_q7 > 0),
            self.effective_ego_scale_q7.saturating_add(8).min(96),
            self.effective_ego_scale_q7,
        );

        self.effective_ego_scale_q7 = branchless_select_u8(spike, damped_scale, healthy_recover);
        self.excitation_potential_u8 = branchless_select_u8(spike, 0, self.excitation_potential_u8);
        self.suppression_cooldown_steps = branchless_select_u8(spike, 3, cooled);
        self.state_code = branchless_select_u8(
            spike,
            DistortionTensionStateCode::ActiveImmediateDamping as u8,
            DistortionTensionStateCode::BalancedDeadband as u8,
        );
        self.immediate_damping_events = self
            .immediate_damping_events
            .saturating_add(u32::from(spike));
    }

    /// Executes Slow-Loop tension excitation on `PassiveSaturation` / chatter stagnation (`0 fdiv`).
    ///
    /// Accumulates `excitation_potential_u8` gradually and raises `effective_ego_scale_q7` only
    /// when Hot-Path suppression cooldown has cleared.
    #[inline]
    pub fn step_slow_loop_excitation(
        &mut self,
        passive_saturation: bool,
        chatter_potential_u8: u8,
        ego_enabled: bool,
    ) {
        let can_excite = ego_enabled
            & (self.suppression_cooldown_steps == 0)
            & (passive_saturation | (chatter_potential_u8 >= 32));
        if !can_excite {
            return;
        }
        let next_pot = self
            .excitation_potential_u8
            .saturating_add(16u8.saturating_add(chatter_potential_u8 >> 3));
        if next_pot >= 96 {
            self.excitation_potential_u8 = next_pot.saturating_sub(64);
            self.effective_ego_scale_q7 = self
                .effective_ego_scale_q7
                .max(32)
                .saturating_add(12)
                .min(112);
            self.state_code = DistortionTensionStateCode::PassiveExcitation as u8;
            self.slow_excitation_events = self.slow_excitation_events.saturating_add(1);
        } else {
            self.excitation_potential_u8 = next_pot;
        }
    }
}

/// Classification of cross-layer geometric/control distortion relative to Layer 2 Ego alignment.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerDistortionKind {
    /// Core layers and ActionHub remain balanced and aligned (`0`).
    #[default]
    Aligned = 0,
    /// Passive saturation or drift without sufficient Ego corrective intervention (`1`).
    PassiveSaturation = 1,
    /// Excessive or unaligned Ego overdrive causing saturation or conflict mismatch (`2`).
    ActiveOverdrive = 2,
}

/// Lazy-computed diagnostic summary of cross-layer distortion (`Distortion`) used to trigger
/// corrective Ego awakening on passive stalls or Ego cooldown on active overdrive.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LayerDistortionReport {
    /// Largest decision share among the 7 `ActionHub` regimes (`0.0..=1.0`).
    pub action_saturation_ratio: f32,
    /// Passive distortion score combining unguided stalls and single-mode saturation (`0.0..=1.0`).
    pub passive_distortion_ratio: f32,
    /// Active overdrive distortion score combining unaligned Ego steps and saturated Ego bias (`0.0..=1.0`).
    pub active_distortion_ratio: f32,
    /// Overall cross-layer mismatch rate (`0.0..=1.0`).
    pub mismatch_ratio: f32,
    /// Dominant distortion state.
    pub kind: LayerDistortionKind,
    /// `true` when passive distortion warrants waking/boosting Layer 2 Ego.
    pub ego_wakeup_recommended: bool,
    /// `true` when active distortion warrants cooling down Layer 2 Ego thrust/defiance.
    pub ego_cooldown_recommended: bool,
}

// SAFETY: `DistortionDelta` is `#[repr(C, align(64))]`, exact size 128 bytes with zero padding.
unsafe impl Pod for DistortionDelta {}
// SAFETY: `DistortionTension` is `#[repr(C, align(16))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for DistortionTension {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distortion_tension_asymmetric_homeostasis() {
        let mut t = DistortionTension::new(true);
        // Immediate 1-step Hot-Path damping under overdrive / guard proximity.
        t.step_hot_path_branchless(80, 64, 90, 98, true);
        assert_eq!(t.effective_ego_scale_q7, 0);
        assert_eq!(t.immediate_damping_events, 1);
        assert_eq!(t.suppression_cooldown_steps, 3);
        assert_eq!(
            t.state_code,
            DistortionTensionStateCode::ActiveImmediateDamping as u8
        );

        // While cooldown > 0, Slow-Loop excitation does not override safety interlock.
        t.step_slow_loop_excitation(true, 96, true);
        assert_eq!(t.slow_excitation_events, 0);

        // After cooldown drains, Slow-Loop excitation awakens Ego out of passive stagnation.
        for _ in 0..3 {
            t.step_hot_path_branchless(10, 48, 20, 0, false);
        }
        for _ in 0..6 {
            t.step_slow_loop_excitation(true, 96, true);
        }
        assert!(t.slow_excitation_events >= 1);
        assert!(t.effective_ego_scale_q7 >= 32);
    }
}
