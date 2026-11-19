// name: src/core/src/traits/gateway/distortion.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `2 Primitives x 2 Drivers` Cognitive Architecture, 4-Region Branchless Venn Partition,
//! [`SystemSpineState`] Physical Pin Isolation, Directed [`DistortionDelta`] P-Frame Chain
//! ([`EgoDifferentialNet`]), and Asymmetric [`DistortionTension`] Homeostasis.
//!
//! Architectural separation (`2 Primitives x 2 Drivers`):
//! - **Substrates ([`crate::traits::primitives::SubstrateKind`])**:
//!   - `Liminal`: 8-granule Venn 4-region mapping ($S \cap O$, $S \setminus O$, $O \setminus S$,
//!     $\overline{S \cup O}$) and $4p(1-p)$ boundary sensitivity (`LiminalVennPartition`).
//!   - `Stigmergy`: 8-granule quorum, noise deadband tolerances, and volatile trace decay.
//! - **Drivers ([`crate::traits::primitives::DriverKind`])**:
//!   - `System` ([`SystemSpineState`]): deterministic I-Frame optimization trajectory $f_{\mathrm{sys}}$,
//!     physical `pinned` isolation against Ego masks, pre-inference bypass, and gauge re-anchoring.
//!   - `Ego` ([`EgoDifferentialNet`] + [`DistortionDelta`] + [`DistortionTension`]): anisotropic
//!     differential P-Frame residual network relative to $f_{\mathrm{sys}}$, governed by asymmetric
//!     homeostatic tension (1-step hot-path damping on overdrive vs. slow-loop excitation on stagnation).

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use super::core_layers::{EgoIntentAxes, LiminalEgoField, StigmergicGranuleField};
use super::epistemic::LiminalApertureField;
use super::*;
use crate::crystal::CrystallizedAttractorRecord;
use crate::traits::primitives::{
    ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE,
    AttributionTag, MAX_ATTENTION_HEADS, SWAR_SPLAT_BYTE_64, branchless_select_u8,
    branchless_select_u32, pack_action_slot_tag, swar_q7_circular_add_x8,
    swar_q7_mean_i8_from_words_64, swar_q7_scale_le128_with_bias_x8,
};
use crate::traits::telemetry::NUM_STIGMERGIC_GRANULES;

/// Maximum number of inline P-Frame [`DistortionDelta`] nodes hosted by [`EgoDifferentialNet`].
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

/// Venn region bit `0`: trigger on $S \cap O$ (`shared_core_mask`).
pub const VENN_REGION_SHARED_CORE: u8 = 1 << 0;

/// Venn region bit `1`: trigger on $S \setminus O$ (`self_exclusive_mask`).
pub const VENN_REGION_SELF_EXCLUSIVE: u8 = 1 << 1;

/// Venn region bit `2`: trigger on validated $O \setminus S$ (`known_unknown_mask`).
pub const VENN_REGION_KNOWN_UNKNOWN: u8 = 1 << 2;

// ===================================================================
// 1. Unified 64-Byte Cognitive Substrate (`Liminal` + `Stigmergy`)
// ===================================================================

/// Unified single-cacheline (`64 B`, `#[repr(C, align(64))]`) cognitive substrate pairing the two
/// foundational geometric primitives: [`LiminalApertureField`] (`0..32`) and
/// [`StigmergicGranuleField`] (`32..64`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CognitiveSubstrate {
    /// Boundary apertures, cognitive window floors, and domain gates (`offset 0..32`).
    pub liminal: LiminalApertureField,
    /// 8-granule stigmergic step gains, noise tolerances, and volatile pheromones (`offset 32..64`).
    pub stigmergy: StigmergicGranuleField,
}

impl CognitiveSubstrate {
    /// Constructs a neutral 64-byte [`CognitiveSubstrate`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal: LiminalApertureField::new(),
            stigmergy: StigmergicGranuleField::new(),
        }
    }
}

// ===================================================================
// 2. 4-Region Branchless Venn Partition (`LiminalVennPartition`, 16 B)
// ===================================================================

/// Branchless 16-byte (`#[repr(C, align(16))]`) 4-region Venn partition across 8 SWAR granules
/// separating $S \cap O$ (shared ground), $S \setminus O$ (self-exclusive gauge origin),
/// validated $O \setminus S$ (known unknown eligible only for non-copy sandbox routing), and
/// $\overline{S \cup O}$ (True Void structural aperture 0).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LiminalVennPartition {
    /// 8-granule cognitive boundary bitmask $S$ of self/active memory (`offset 0`).
    pub self_mask: u8,
    /// 8-granule noise-filtered external bitmask $O$ (`density > noise_tolerance`, `offset 1`).
    pub other_mask: u8,
    /// $S \cap O$: shared common ground eligible for steady shortcut routing (`offset 2`).
    pub shared_core_mask: u8,
    /// $S \setminus O$: protected self-specific domain / gauge origin (`offset 3`).
    pub self_exclusive_mask: u8,
    /// Validated $O \setminus S$: known unknown routed strictly to isolated sandbox (`offset 4`).
    pub known_unknown_mask: u8,
    /// $\overline{S \cup O}$ (plus rejected isolated single-granule noise): True Void (`offset 5`).
    pub true_void_mask: u8,
    /// Number of $S \cap O$ bridge granules (`popcnt(shared_core_mask)`, `offset 6`).
    pub bridge_count: u8,
    /// Local concentration in Q0.7 (`0..=127`) across exterior granules (`offset 7`).
    pub other_concentration_q7: u8,
    /// Branchless $4p(1-p)$ boundary sensitivity in Q0.7 (`0..=127`, `0 fdiv`, `offset 8`).
    pub boundary_sensitivity_q7: u8,
    /// Mean excess Q0.7 density across validated $O \setminus S$ granules (`offset 9`).
    pub known_unknown_density_q7: u8,
    /// `1` when validated $O \setminus S$ exterior pointers require isolated sandbox staging (`offset 10`).
    pub sandbox_eligible: u8,
    /// `1` when both $S \cap O$ and validated $O \setminus S$ are empty (aperture 0 True Void lock, `offset 11`).
    pub true_void_aperture_zero: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`, `System` or `Ego`, `offset 12`).
    pub attribution_tag: u8,
    /// Explicit C alignment padding (`offset 13..16`).
    pub reserved_align: [u8; 3],
}

/// Evaluates the branchless 4-region Venn partition across 8 SWAR granules (`0 alloc`, `0 fdiv`,
/// `0` data-dependent branches).
///
/// Validation rule for $O \setminus S$ (Known Unknown):
/// - Each granule `g` joins `other_mask` only if `other_granule_density_q7[g] > granule_noise_tolerance_q7[g]`.
/// - Unbridged `O \ S` (`bridge_count == 0`) requires at least `2` active exterior granules AND
///   `other_concentration_q7 >= concentration_floor_q7`; otherwise isolated single-granule noise
///   drops into `true_void_mask` ($\overline{S \cup O}$) with `known_unknown_mask = 0`.
#[inline]
#[must_use]
pub fn evaluate_liminal_venn_partition_branchless(
    self_mask: u8,
    other_granule_density_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    granule_noise_tolerance_q7: &[u8; NUM_STIGMERGIC_GRANULES],
    concentration_floor_q7: u8,
) -> LiminalVennPartition {
    let mut other_mask = 0u8;
    let mut g = 0usize;
    while g < NUM_STIGMERGIC_GRANULES {
        let bit = 1u8 << g;
        let above = other_granule_density_q7[g] > granule_noise_tolerance_q7[g];
        other_mask |= branchless_select_u8(above, bit, 0);
        g += 1;
    }

    let shared_core_mask = self_mask & other_mask;
    let self_exclusive_mask = self_mask & !other_mask;
    let raw_ext_mask = !self_mask & other_mask;
    let bridge_count = shared_core_mask.count_ones() as u8;
    let ext_cnt = raw_ext_mask.count_ones() as u8;

    let mut peak_excess = 0u8;
    let mut sum_excess = 0u16;
    let mut g2 = 0usize;
    while g2 < NUM_STIGMERGIC_GRANULES {
        let in_ext = ((raw_ext_mask >> g2) & 1) != 0;
        let ex = other_granule_density_q7[g2].saturating_sub(granule_noise_tolerance_q7[g2]);
        let eff_ex = branchless_select_u8(in_ext, ex, 0);
        peak_excess = branchless_select_u8(eff_ex > peak_excess, eff_ex, peak_excess);
        sum_excess = sum_excess.wrapping_add(u16::from(eff_ex));
        g2 += 1;
    }

    // Branchless fixed reciprocal multiply by `1/k` scaled by 128 (`>> 7`) to avoid hardware `div`
    const INV_CNT_Q7: [u16; 9] = [0, 128, 64, 43, 32, 26, 21, 18, 16];
    let mean_ext_excess =
        (((sum_excess * INV_CNT_Q7[usize::from(ext_cnt.min(8))]) >> 7).min(127)) as u8;

    let unbridged_conc = ((sum_excess >> 1).min(127) as u8).max(peak_excess.min(127));
    let other_concentration_q7 = branchless_select_u8(
        bridge_count >= 1,
        peak_excess.min(127).max(mean_ext_excess),
        branchless_select_u8(ext_cnt >= 2, unbridged_conc, 0),
    );

    let ext_valid = (raw_ext_mask != 0)
        & ((bridge_count >= 1) | (other_concentration_q7 >= concentration_floor_q7));
    let known_unknown_mask = branchless_select_u8(ext_valid, raw_ext_mask, 0);
    let known_unknown_density_q7 = branchless_select_u8(ext_valid, mean_ext_excess, 0);

    let true_void_mask = !(shared_core_mask | self_exclusive_mask | known_unknown_mask);

    // 4p(1-p) in Q0.7 (`0 fdiv`): p_q7 = active_cnt * 16 in [0..=128], sensitivity = (p_q7 * (128 - p_q7)) >> 5
    let active_other_cnt = (shared_core_mask | known_unknown_mask).count_ones();
    let p_q7 = (active_other_cnt << 4).min(128);
    let sens_raw = (p_q7.saturating_mul(128u32.saturating_sub(p_q7))) >> 5;
    let boundary_sensitivity_q7 = sens_raw.min(127) as u8;

    let sandbox_eligible = u8::from(known_unknown_mask != 0);
    let true_void_aperture_zero = u8::from((shared_core_mask == 0) & (known_unknown_mask == 0));

    LiminalVennPartition {
        self_mask,
        other_mask,
        shared_core_mask,
        self_exclusive_mask,
        known_unknown_mask,
        true_void_mask,
        bridge_count,
        other_concentration_q7,
        boundary_sensitivity_q7,
        known_unknown_density_q7,
        sandbox_eligible,
        true_void_aperture_zero,
        attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
        reserved_align: [0; 3],
    }
}

// ===================================================================
// 3. System Spine State & Physical `pinned` Isolation (32 B)
// ===================================================================

/// Mechanical execution spine (`32 B`, `#[repr(C, align(16))]`, `DriverKind::System` = I-Frame).
///
/// Owns physical `pinned` masks isolating irreversible steps, memory safety, OOD aperture-0 locks,
/// and explicit contradiction guards so [`EgoDifferentialNet`] masks/XORs cannot reach them.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemSpineState {
    /// Physical `Liminal` aperture bits protected from Ego override (`offset 0..4`).
    pub pinned_aperture_mask: u32,
    /// Physical workbench slot bits protected from Ego deflection (`offset 4`).
    pub pinned_slot_mask: u8,
    /// Physical `ACTION_MODE_*` bits protected from Ego suppression (`offset 5`).
    pub pinned_action_mode_mask: u8,
    /// Non-zero when True Void ($\overline{S \cup O}$) forces aperture-0 lock (`offset 6`).
    pub ood_void_locked: u8,
    /// Non-zero during irreversible Prefill / safety checkpoints (`offset 7`).
    pub irreversible_step_lock: u8,
    /// Physical compute slot bitmask protected from Fast-Loop bypass (`offset 8..16`).
    pub pinned_compute_mask: u64,
    /// Monotonic gauge re-anchoring I-Frame epoch counter (`offset 16..20`).
    pub gauge_anchor_epoch: u32,
    /// Monotonic Slow-Loop epoch counter (`offset 20..24`).
    pub slow_loop_epoch: u32,
    /// Non-zero when token-level whole-steering bypass is active (`0` D2H syncs, `offset 24`).
    pub whole_token_bypass_active: u8,
    /// Consecutive calm steps accumulated toward pre-inference bypass (`offset 25`).
    pub bypass_dwell_steps: u8,
    /// Remaining recovery cooldown steps before bypass may re-engage (`offset 26`).
    pub bypass_cooldown_steps: u8,
    /// 2-bit [`AttributionTag`] (`Liminal`/`System`, `offset 27`).
    pub attribution_tag: u8,
    /// Explicit C alignment padding (`offset 28..32`).
    pub reserved_align: [u8; 4],
}

impl Default for SystemSpineState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpineState {
    /// Constructs a canonical [`SystemSpineState`] locking SystemConstraint, ContradictionGuard,
    /// and `ACTION_MODE_PHASE_REPULSE` in physical `pinned` storage.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pinned_aperture_mask: APERTURE_SYSTEM_CONSTRAINT | APERTURE_CONTRADICTION_GUARD,
            pinned_slot_mask: (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                | (1u8 << ATTRACTOR_SLOT_REPULSION),
            pinned_action_mode_mask: crate::traits::primitives::ACTION_MODE_PHASE_REPULSE,
            ood_void_locked: 0,
            irreversible_step_lock: 0,
            pinned_compute_mask: 0,
            gauge_anchor_epoch: 0,
            slow_loop_epoch: 0,
            whole_token_bypass_active: 0,
            bypass_dwell_steps: 0,
            bypass_cooldown_steps: 0,
            attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
            reserved_align: [0; 4],
        }
    }

    /// Physical `pinned` isolation barrier ensuring Ego modulation never mutates pinned apertures.
    #[inline(always)]
    #[must_use]
    pub const fn mask_ego_aperture_edits(
        &self,
        prev_aperture_mask: u32,
        next_aperture_mask: u32,
    ) -> u32 {
        (next_aperture_mask & !self.pinned_aperture_mask)
            | (prev_aperture_mask & self.pinned_aperture_mask)
    }

    /// Physical `pinned` isolation barrier executed without branching after Ego modulates `ego_frame`.
    ///
    /// - Under `ood_void_locked != 0` or `irreversible_step_lock != 0`, returns `*baseline_frame`.
    /// - Always preserves every `pinned_action_mode_mask` bit from `baseline_frame` and blocks
    ///   slot overrides away from any `pinned_slot_mask` slot.
    #[inline]
    #[must_use]
    pub fn enforce_pinned_isolation_branchless(
        &self,
        baseline_frame: &GatewayFrame,
        mut ego_frame: GatewayFrame,
    ) -> GatewayFrame {
        let base_slot = baseline_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK;
        let slot_is_pinned = (self.pinned_slot_mask & (1u8 << base_slot)) != 0;
        ego_frame.dominant_slot =
            branchless_select_u8(slot_is_pinned, baseline_frame.dominant_slot, ego_frame.dominant_slot);

        let preserved_modes = baseline_frame.action_mode_flags & self.pinned_action_mode_mask;
        ego_frame.action_mode_flags |= preserved_modes;

        let full_lock = (self.ood_void_locked != 0) | (self.irreversible_step_lock != 0);
        if full_lock {
            return *baseline_frame;
        }

        let ptr_tag = pack_action_slot_tag(
            ego_frame.dominant_slot & ATTRACTOR_SLOT_ID_MASK,
            ego_frame.action_mode_flags,
        );
        ego_frame.frame_flags = u16::from(ptr_tag);
        ego_frame.tagged_ptr = gcso_tagged_ptr_t::pack(
            ego_frame.mapped_section_offset48,
            ptr_tag,
            ego_frame.mean_phase_tag_q7,
        )
        .raw();
        ego_frame
    }
}

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
    /// Bitmask of allowed Venn regions (`bit 0=S∩O`, `bit 1=S\O`, `bit 2=O\S`, `offset 12`).
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
            venn_region_selector: VENN_REGION_SHARED_CORE | VENN_REGION_SELF_EXCLUSIVE,
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
/// between [`SystemSpineState`] convergence and [`EgoDifferentialNet`] deviation.
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
        let mild_floor =
            branchless_select_u8(self.effective_ego_scale_q7 > 0, quarter.max(8), 0);
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
        let can_excite =
            ego_enabled & (self.suppression_cooldown_steps == 0) & (passive_saturation | (chatter_potential_u8 >= 32));
        if !can_excite {
            return;
        }
        let next_pot = self
            .excitation_potential_u8
            .saturating_add(16u8.saturating_add(chatter_potential_u8 >> 3));
        if next_pot >= 96 {
            self.excitation_potential_u8 = next_pot.saturating_sub(64);
            self.effective_ego_scale_q7 = self.effective_ego_scale_q7.max(32).saturating_add(12).min(112);
            self.state_code = DistortionTensionStateCode::PassiveExcitation as u8;
            self.slow_excitation_events = self.slow_excitation_events.saturating_add(1);
        } else {
            self.excitation_potential_u8 = next_pot;
        }
    }
}

// ===================================================================
// 6. Non-Parametric Residual P-Frame Chain (`EgoDifferentialNet`)
// ===================================================================

/// Inline fixed-capacity (`8` nodes, `0` heap allocations) non-parametric P-Frame residual
/// neural network evaluating directed [`DistortionDelta`] pointer chains (up to `max_hops`
/// in `1..=4`) triggered by Venn region support and stigmergic density.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgoDifferentialNet {
    /// Fixed array of up to 8 directed P-Frame [`DistortionDelta`] nodes.
    pub nodes: [DistortionDelta; MAX_EGO_PFRAME_NODES],
    /// 8-bit bitmask marking populated nodes (`0..=7`).
    pub active_node_mask: u8,
    /// Maximum hop traversal bound (`1..=4`, allows clean 1-hop vs multi-hop comparison).
    pub max_hops: u8,
    /// Minimum hop depth before Cold-Path GOP consolidation crystallizes an I-Frame anchor.
    pub consolidation_hop_limit: u8,
    /// Stigmergic pheromone floor (`0..=127`) below which unused P-Frame nodes evaporate.
    pub evaporation_floor_q7: u8,
    /// Hop count traversed on the most recent evaluation.
    pub last_hops_traversed: u8,
    /// Mean differential magnitude (`0..=127`) observed on the most recent evaluation.
    pub last_delta_norm_q7: u8,
    /// Monotonic Cold-Path maintenance epoch counter.
    pub epoch: u32,
}

impl Default for EgoDifferentialNet {
    #[inline(always)]
    fn default() -> Self {
        Self::new_empty(1)
    }
}

impl EgoDifferentialNet {
    /// Creates an empty inert [`EgoDifferentialNet`] with bounded `max_hops` (`1..=4`).
    #[inline(always)]
    #[must_use]
    pub const fn new_empty(max_hops: u8) -> Self {
        let clamped_hops = if max_hops == 0 {
            1
        } else if max_hops > MAX_DISTORTION_CHAIN_HOPS {
            MAX_DISTORTION_CHAIN_HOPS
        } else {
            max_hops
        };
        Self {
            nodes: [DistortionDelta::new_empty(0, 0); MAX_EGO_PFRAME_NODES],
            active_node_mask: 0,
            max_hops: clamped_hops,
            consolidation_hop_limit: 3,
            evaporation_floor_q7: 16,
            last_hops_traversed: 0,
            last_delta_norm_q7: 0,
            epoch: 0,
        }
    }

    /// Deterministically initializes a 3-hop P-Frame chain from [`LiminalEgoField`] and [`EgoIntentAxes`]:
    /// - Node 0 (`S ∩ O`): persona heading alignment delta
    /// - Node 1 (`S \ O`): sovereign counter-basin stylistic delta
    /// - Node 2 (`O \ S`): transgressive exterior bridge delta
    #[inline]
    #[must_use]
    pub fn from_ego_field_and_axes(
        ego: &LiminalEgoField,
        axes: &EgoIntentAxes,
        max_hops: u8,
    ) -> Self {
        let mut net = Self::new_empty(max_hops);
        if !ego.is_active() {
            return net;
        }

        let make_phase = |base: &[gcso_q7_t; NUM_STIGMERGIC_GRANULES], shift: i8, flip_odd: bool| -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
            let mut out = [0 as gcso_q7_t; MAX_ATTENTION_HEADS];
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                let gval = i32::from(base[g]);
                let dir = if flip_odd && ((g & 1) == 1) { -gval } else { gval };
                let s = (dir + i32::from(shift)).clamp(-127, 127) as i8;
                let base_h = g << 3;
                let mut k = 0usize;
                while k < 8 {
                    let mod_off = ((k as i32) - 3) * (i32::from(shift) >> 4);
                    out[base_h + k] = (i32::from(s) + mod_off).clamp(-127, 127) as gcso_q7_t;
                    k += 1;
                }
                g += 1;
            }
            out
        };

        let mut n0 = DistortionDelta::new_empty(1, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 1);
        n0.next_hop_idx = 1;
        n0.hop_depth = 1;
        n0.venn_trigger_mask = if axes.active_granule_bias_mask == 0 {
            0xFF
        } else {
            axes.active_granule_bias_mask
        };
        n0.venn_region_selector = VENN_REGION_SHARED_CORE;
        n0.pheromone_q7 = 88;
        n0.gain_q7 = ego.counter_basin_thrust_q7.clamp(32, 127);
        n0.intent_delta = *axes;
        n0.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n0.delta_phase_q7 = make_phase(&ego.sovereign_granule_phase_q7, 4, false);

        let mut n1 = DistortionDelta::new_empty(2, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 2);
        n1.next_hop_idx = 2;
        n1.hop_depth = 2;
        n1.venn_trigger_mask = 0xAA;
        n1.venn_region_selector = VENN_REGION_SHARED_CORE | VENN_REGION_SELF_EXCLUSIVE;
        n1.pheromone_q7 = 76;
        n1.gain_q7 = ego.irrational_defiance_q7.clamp(24, 127);
        n1.intent_delta = *axes;
        n1.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n1.delta_phase_q7 = make_phase(&ego.sovereign_granule_phase_q7, ego.transgression_bias_q7 >> 2, true);

        let mut n2 = DistortionDelta::new_empty(3, u32::from(ATTRACTOR_SLOT_SYSTEM_BASELINE) + 3);
        n2.next_hop_idx = DISTORTION_DELTA_NO_NEXT_HOP;
        n2.hop_depth = 3;
        n2.venn_trigger_mask = 0x55;
        n2.venn_region_selector = VENN_REGION_KNOWN_UNKNOWN | VENN_REGION_SELF_EXCLUSIVE | VENN_REGION_SHARED_CORE;
        n2.pheromone_q7 = 64;
        n2.gain_q7 = ego.transgression_bias_q7.unsigned_abs().clamp(24, 127);
        n2.intent_delta = *axes;
        n2.flags = DISTORTION_DELTA_FLAG_ACTIVE | DISTORTION_DELTA_FLAG_BAKED_PHASE_VALID;
        n2.delta_phase_q7 = make_phase(&ego.sovereign_granule_phase_q7, -(ego.transgression_bias_q7 >> 2), false);

        net.nodes[0] = n0;
        net.nodes[1] = n1;
        net.nodes[2] = n2;
        net.active_node_mask = 0b0000_0111;
        net
    }

    /// Evaluates the fixed-length [`DistortionDelta`] chain (unrolled up to 4 hops, masked by
    /// `self.max_hops`, `0` heap alloc, `0 fdiv`, `0` data-dependent inner branches) and returns
    /// `(modulated_frame, delta_norm_q7, hops_traversed)`.
    ///
    /// When Ego is inactive, tension scale is 0, or `spine` enforces `ood_void_locked` /
    /// `irreversible_step_lock`, returns `(base_frame, 0, 0)` (zero-cost unobservable bypass).
    #[inline]
    #[must_use]
    pub fn evaluate_pframe_chain_branchless(
        &self,
        ego: &LiminalEgoField,
        spine: &SystemSpineState,
        venn: &LiminalVennPartition,
        stigmergy: &StigmergicGranuleField,
        tension: &DistortionTension,
        base_frame: GatewayFrame,
    ) -> (GatewayFrame, u8, u8) {
        let (frame, norm, hops, _entry) = self.evaluate_pframe_chain_with_entry_branchless(
            ego, spine, venn, stigmergy, tension, base_frame,
        );
        (frame, norm, hops)
    }

    /// Evaluates the fixed-length [`DistortionDelta`] chain and additionally returns the selected
    /// entry node index `entry_idx` (`0..8`) for reinforcement telemetry tracking.
    #[inline]
    #[must_use]
    pub fn evaluate_pframe_chain_with_entry_branchless(
        &self,
        ego: &LiminalEgoField,
        spine: &SystemSpineState,
        venn: &LiminalVennPartition,
        stigmergy: &StigmergicGranuleField,
        tension: &DistortionTension,
        base_frame: GatewayFrame,
    ) -> (GatewayFrame, u8, u8, u8) {
        if !ego.is_active()
            || tension.effective_ego_scale_q7 == 0
            || spine.ood_void_locked != 0
            || spine.irreversible_step_lock != 0
            || self.active_node_mask == 0
        {
            return (base_frame, 0, 0, 0);
        }

        let region_bits = branchless_select_u8(venn.shared_core_mask != 0, VENN_REGION_SHARED_CORE, 0)
            | branchless_select_u8(venn.self_exclusive_mask != 0, VENN_REGION_SELF_EXCLUSIVE, 0)
            | branchless_select_u8(venn.known_unknown_mask != 0, VENN_REGION_KNOWN_UNKNOWN, 0);
        let region_fallback = branchless_select_u8(
            region_bits != 0,
            region_bits,
            VENN_REGION_SHARED_CORE | VENN_REGION_SELF_EXCLUSIVE,
        );
        let combined_support =
            venn.shared_core_mask | venn.self_exclusive_mask | venn.known_unknown_mask;
        let support_mask = branchless_select_u8(combined_support != 0, combined_support, 0xFF);

        // Select strongest matching entry node across the 8 fixed slots branchlessly.
        let mut best_entry_idx = 0usize;
        let mut best_entry_score = 0u32;
        let mut i = 0usize;
        while i < MAX_EGO_PFRAME_NODES {
            let n = &self.nodes[i];
            let active = ((self.active_node_mask >> i) & 1) != 0 && n.is_baked_active();
            let reg_ok = (n.venn_region_selector & region_fallback) != 0;
            let trig_overlap = (n.venn_trigger_mask & support_mask).count_ones();
            let score = if active && reg_ok && trig_overlap > 0 {
                u32::from(n.pheromone_q7).saturating_mul(trig_overlap)
            } else {
                0
            };
            let take = score > best_entry_score;
            best_entry_score = branchless_select_u32(take, score, best_entry_score);
            best_entry_idx = if take { i } else { best_entry_idx };
            i += 1;
        }
        if best_entry_score == 0 {
            return (base_frame, 0, 0, 0);
        }

        // Fixed 4-hop unrolled P-Frame accumulation bounded by `self.max_hops`.
        let hop_cap = self.max_hops.min(MAX_DISTORTION_CHAIN_HOPS);
        let sens_boost_q7 = u32::from(venn.boundary_sensitivity_q7).max(48);
        let eff_scale = u32::from(tension.effective_ego_scale_q7);
        let mean_stig_ph = {
            let mut s = 0u32;
            let mut g = 0usize;
            while g < NUM_STIGMERGIC_GRANULES {
                s += u32::from(stigmergy.granule_pheromone_q7[g]);
                g += 1;
            }
            (s >> 3).clamp(24, 127)
        };

        let mut acc_words = *base_frame.phase_words_ref();
        let mut accum_norm_sum = 0u32;
        let mut hops_traversed = 0u8;
        let mut cur_idx = best_entry_idx as u16;
        let mut hop = 0u8;

        while hop < MAX_DISTORTION_CHAIN_HOPS {
            let in_bounds = (hop < hop_cap)
                && (usize::from(cur_idx) < MAX_EGO_PFRAME_NODES)
                && (((self.active_node_mask >> (cur_idx & 7)) & 1) != 0);
            if !in_bounds {
                break;
            }
            let node = &self.nodes[usize::from(cur_idx)];
            if !node.is_baked_active() {
                break;
            }
            let hop_weight_q7 =
                (((u32::from(node.gain_q7) * eff_scale) >> 7) * sens_boost_q7) >> 7;
            let dens_weight_q7 = ((hop_weight_q7 * mean_stig_ph) >> 7).clamp(8, 127) as u64;
            let bias_restore = (128u64.wrapping_sub(dens_weight_q7)).wrapping_mul(SWAR_SPLAT_BYTE_64);
            let node_words = crate::traits::primitives::load_q7_words_64(&node.delta_phase_q7);
            let mut c = 0usize;
            while c < NUM_STIGMERGIC_GRANULES {
                let d_w = swar_q7_scale_le128_with_bias_x8(node_words[c], dens_weight_q7, bias_restore);
                acc_words[c] = swar_q7_circular_add_x8(acc_words[c], d_w);
                c += 1;
            }
            accum_norm_sum = accum_norm_sum.saturating_add(dens_weight_q7 as u32);
            hops_traversed = hops_traversed.saturating_add(1);
            cur_idx = node.next_hop_idx;
            hop = hop.saturating_add(1);
        }

        let mut ego_frame = base_frame;
        *ego_frame.phase_words_mut() = acc_words;
        ego_frame.mean_phase_tag_q7 = swar_q7_mean_i8_from_words_64(&acc_words);

        let isolated = spine.enforce_pinned_isolation_branchless(&base_frame, ego_frame);
        const INV_HOPS_Q7: [u32; 5] = [0, 128, 64, 43, 32];
        let avg_delta =
            ((accum_norm_sum * INV_HOPS_Q7[usize::from(hops_traversed.min(4))]) >> 7).min(127)
                as u8;
        (isolated, avg_delta, hops_traversed, best_entry_idx as u8)
    }

    /// Records positive traversal along up to `hops_traversed` chain hops starting at `entry_idx`,
    /// advancing `reinforce_count` and stamping `last_active_epoch = self.epoch` for Cold-Path GOP
    /// consolidation and evaporation pruning.
    #[inline]
    pub fn record_chain_activation(&mut self, entry_idx: usize, hops_traversed: u8) {
        self.last_hops_traversed = hops_traversed;
        let mut cur = entry_idx;
        let mut h = 0u8;
        let cap = hops_traversed.min(MAX_DISTORTION_CHAIN_HOPS);
        while h < cap && cur < MAX_EGO_PFRAME_NODES {
            if (self.active_node_mask & (1u8 << cur)) == 0 {
                break;
            }
            let node = &mut self.nodes[cur];
            node.reinforce_count = node.reinforce_count.saturating_add(1);
            node.last_active_epoch = self.epoch;
            cur = usize::from(node.next_hop_idx);
            h = h.saturating_add(1);
        }
    }

    /// Cold-Path GOP maintenance: consolidates reinforced multi-hop P-Frame chains into an I-Frame
    /// [`CrystallizedAttractorRecord`] and evaporates stale unreinforced P-Frame nodes below `evaporation_floor_q7`.
    #[inline]
    pub fn consolidate_and_evaporate_cold_path(
        &mut self,
        current_epoch: u32,
    ) -> Option<CrystallizedAttractorRecord> {
        self.epoch = current_epoch;
        let mut crystallized: Option<CrystallizedAttractorRecord> = None;

        for idx in 0..MAX_EGO_PFRAME_NODES {
            let bit = 1u8 << idx;
            if (self.active_node_mask & bit) == 0 {
                continue;
            }
            let node = &mut self.nodes[idx];
            if !node.is_active() {
                self.active_node_mask &= !bit;
                continue;
            }
            if node.hop_depth >= self.consolidation_hop_limit
                && node.reinforce_count >= 4
                && (node.flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME) == 0
            {
                let mut rec = CrystallizedAttractorRecord::new_empty(
                    node.anchor_ref_id,
                    ATTRACTOR_SLOT_SYSTEM_BASELINE,
                    0,
                );
                rec.phase_profile = node.delta_phase_q7;
                rec.base_energy = f32::from(node.gain_q7) * (1.0 / 127.0);
                rec.sample_count = node.reinforce_count;
                rec.stamp_crc32();
                node.flags |= DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME;
                crystallized = Some(rec);
            }

            if current_epoch.saturating_sub(node.last_active_epoch) >= 4 {
                let decay = (node.pheromone_q7 >> 3).max(1);
                node.pheromone_q7 = node.pheromone_q7.saturating_sub(decay);
                if node.pheromone_q7 < self.evaporation_floor_q7 {
                    node.flags &= !DISTORTION_DELTA_FLAG_ACTIVE;
                    self.active_node_mask &= !bit;
                }
            }
        }
        crystallized
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::gateway::LiminalEgoPreset;
    use crate::traits::primitives::{ACTION_MODE_DPSR_STEER, ACTION_MODE_PHASE_REPULSE};

    #[test]
    fn test_liminal_venn_partition_four_regions_and_noise_deadband() {
        let noise_tol = [10u8; NUM_STIGMERGIC_GRANULES];

        // 1. Isolated single-granule OOD noise collapses to True Void (`known_unknown_mask == 0`).
        let mut single_noise = [0u8; NUM_STIGMERGIC_GRANULES];
        single_noise[5] = 28;
        let vp_noise = evaluate_liminal_venn_partition_branchless(0x03, &single_noise, &noise_tol, 40);
        assert_eq!(vp_noise.shared_core_mask, 0);
        assert_eq!(vp_noise.self_exclusive_mask, 0x03);
        assert_eq!(vp_noise.known_unknown_mask, 0);
        assert_eq!(vp_noise.true_void_mask, !0x03);
        assert_eq!(vp_noise.true_void_aperture_zero, 1);
        assert_eq!(vp_noise.sandbox_eligible, 0);
        assert_eq!(vp_noise.boundary_sensitivity_q7, 0);

        // 2. Bridged `S ∩ O` + `O \ S` stages validated Known Unknown into sandbox without copy.
        let bridged = [80, 72, 0, 0, 92, 88, 0, 0];
        let vp_ext = evaluate_liminal_venn_partition_branchless(0x03, &bridged, &noise_tol, 40);
        assert_eq!(vp_ext.shared_core_mask, 0x03);
        assert_eq!(vp_ext.self_exclusive_mask, 0);
        assert_eq!(vp_ext.known_unknown_mask, 0x30);
        assert_eq!(vp_ext.bridge_count, 2);
        assert_eq!(vp_ext.sandbox_eligible, 1);
        assert_eq!(vp_ext.true_void_aperture_zero, 0);
        // 4/8 active granules -> 4 * 0.5 * 0.5 = 1.0 -> 127 in Q0.7.
        assert_eq!(vp_ext.boundary_sensitivity_q7, 127);

        // 3. 8/8 saturation -> p = 1.0 -> 4p(1-p) == 0 boundary sensitivity.
        let full = [90u8; NUM_STIGMERGIC_GRANULES];
        let vp_full = evaluate_liminal_venn_partition_branchless(0xFF, &full, &noise_tol, 40);
        assert_eq!(vp_full.boundary_sensitivity_q7, 0);
    }

    #[test]
    fn test_system_spine_physical_pinned_isolation() {
        let mut spine = SystemSpineState::new();
        let baseline = GatewayFrame {
            dominant_slot: ATTRACTOR_SLOT_REPULSION,
            action_mode_flags: ACTION_MODE_DPSR_STEER | ACTION_MODE_PHASE_REPULSE,
            ..GatewayFrame::default()
        };
        let ego_attempt = GatewayFrame {
            dominant_slot: 1,
            action_mode_flags: ACTION_MODE_DPSR_STEER, // stripped PHASE_REPULSE
            phase_deltas: [25; MAX_ATTENTION_HEADS],
            ..GatewayFrame::default()
        };
        let guarded = spine.enforce_pinned_isolation_branchless(&baseline, ego_attempt);
        assert_eq!(guarded.dominant_slot, ATTRACTOR_SLOT_REPULSION);
        assert_ne!(guarded.action_mode_flags & ACTION_MODE_PHASE_REPULSE, 0);

        spine.ood_void_locked = 1;
        let void_locked = spine.enforce_pinned_isolation_branchless(&baseline, ego_attempt);
        assert_eq!(void_locked.phase_deltas, baseline.phase_deltas);
    }

    #[test]
    fn test_ego_differential_net_hops_consolidation_and_evaporation() {
        let ego = LiminalEgoField::from_preset(LiminalEgoPreset::Medium);
        let axes = EgoIntentAxes::new().with_autonomy_q7(72).with_transgression_q7(48);
        let net_1 = EgoDifferentialNet::from_ego_field_and_axes(&ego, &axes, 1);
        let mut net_3 = EgoDifferentialNet::from_ego_field_and_axes(&ego, &axes, 3);
        let spine = SystemSpineState::new();
        let stig = StigmergicGranuleField::new();
        let tension = DistortionTension::new(true);
        let venn = LiminalVennPartition {
            shared_core_mask: 0x0F,
            known_unknown_mask: 0x10,
            boundary_sensitivity_q7: 96,
            ..LiminalVennPartition::default()
        };
        let base = GatewayFrame::default();

        let (f1, _n1, h1) =
            net_1.evaluate_pframe_chain_branchless(&ego, &spine, &venn, &stig, &tension, base);
        let (f3, _n3, h3, entry) =
            net_3.evaluate_pframe_chain_with_entry_branchless(&ego, &spine, &venn, &stig, &tension, base);
        assert_eq!(h1, 1);
        assert_eq!(h3, 3);
        assert_ne!(f1.phase_deltas, f3.phase_deltas);

        for ep in 1..=5 {
            net_3.epoch = ep;
            net_3.record_chain_activation(usize::from(entry), h3);
        }
        assert!(net_3.nodes[2].reinforce_count >= 4);
        let crystal = net_3.consolidate_and_evaporate_cold_path(6);
        assert!(crystal.is_some());
        assert_ne!(
            net_3.nodes[2].flags & DISTORTION_DELTA_FLAG_CONSOLIDATED_IFRAME,
            0
        );

        // Stale unreinforced node evaporates when untouched across epochs.
        net_3.nodes[0].last_active_epoch = 0;
        net_3.nodes[0].pheromone_q7 = 16;
        let _ = net_3.consolidate_and_evaporate_cold_path(12);
        assert!(!net_3.nodes[0].is_active());
    }

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
