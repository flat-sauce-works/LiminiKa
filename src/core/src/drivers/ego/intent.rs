// name: src/core/src/drivers/ego/intent.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core Stack Slot 2 Ego intent representations ([`LiminalEgoPreset`], [`EgoIntentAxes`] 16 B,
//! and [`LiminalEgoState`] 64 B).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use crate::foundation::{
    compute_crc32_ieee, project_bytes_swar_to_words_64, swar_q7_mean_i8_from_words_64, Pod,
    ACTION_MODE_LAYER_SKIP, ACTION_MODE_PHASE_STEER, ACTION_MODE_SUBHEAD_ROUTE,
};
use crate::liminal::{
    branchless_select_i32, branchless_select_i8, branchless_select_u32, branchless_select_u8,
    gcso_q7_t, MAX_ATTRACTOR_DOMAINS, NUM_LIMINAL_THRESHOLD_CHANNELS, NUM_STIGMERGIC_GRANULES,
};
use crate::stigmergy::{
    ATTRACTOR_SLOT_CRYSTALLIZED, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
    ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_USER_CONTEXT, ATTRACTOR_SLOT_USER_INTENT,
};

/// Ego intervention trigger bit (`0x01`): Counter-basin orbital thrust when attractor pull threatens capture.
pub const EGO_TRIGGER_BASIN_CAPTURE: u8 = 1 << 0;
/// Ego intervention trigger bit (`0x02`): Defiant intervention under internal contradiction (`H^1 != 0`).
pub const EGO_TRIGGER_INTERNAL_CONFLICT: u8 = 1 << 1;
/// Ego intervention trigger bit (`0x04`): Transgressive step when standing at a `Liminal` horizon boundary.
pub const EGO_TRIGGER_LIMINAL_EXTERIOR: u8 = 1 << 2;
/// Ego intervention trigger bit (`0x08`): Boundary damping when temporal boundary velocity oscillates.
pub const EGO_TRIGGER_BOUNDARY_CHATTER: u8 = 1 << 3;
/// Ego intervention trigger bit (`0x10`): Self-initiated perturbation when boundaries stagnate completely.
pub const EGO_TRIGGER_BOUNDARY_STAGNATION: u8 = 1 << 4;
/// Ego intervention trigger bit (`0x20`): Corrective awakening when layer distortion or saturation is detected.
pub const EGO_TRIGGER_LAYER_DISTORTION: u8 = 1 << 5;
/// Default active bitmask enabling all 5 deterministic [`LiminalEgoState`] triggers (`0x1F`).
pub const EGO_TRIGGER_ALL_DEFAULT: u8 = 0x1F;

/// Intensity template preset for [`LiminalEgoState`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiminalEgoPreset {
    /// Passive custom/zero baseline (no autonomous counter-basin or transgressive override).
    #[default]
    Custom = 0,
    /// Subtle individuality: gentle counter-basin resistance and boundary stabilization.
    Low = 1,
    /// Balanced autonomous agency: orbital basin resistance with measured curiosity across boundaries.
    Medium = 2,
    /// Strong willpower & autonomy: vigorous counter-basin thrust and defiant boundary transgression.
    High = 3,
}

impl LiminalEgoPreset {
    /// Decodes a compact `u8` preset code (`0..=3`) into [`LiminalEgoPreset`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Low,
            2 => Self::Medium,
            3 => Self::High,
            _ => Self::Custom,
        }
    }
}

/// 16-byte (`#[repr(C, align(16))]`) normalized intermediate semantic intent axes decoupling
/// surface natural-language parsing, DSL syntax, or embedding projections from the fixed 64-byte
/// Core Stack Slot 2 [`LiminalEgoState`] ABI layout.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EgoIntentAxes {
    /// Self-directed resistance against attractor capture (`-127..=127`, `offset 0`).
    pub autonomy_q7: i8,
    /// Curiosity and willingness to step across `Liminal` boundaries (`-127..=127`, `offset 1`).
    pub transgression_q7: i8,
    /// Expressive warmth and subjective persona commitment (`-127..=127`, `offset 2`).
    pub devotion_q7: i8,
    /// Analytical rigor and factual coherence priority (`-127..=127`, `offset 3`).
    pub logic_q7: i8,
    /// Skepticism, guard vigilance, and boundary chatter damping (`-127..=127`, `offset 4`).
    pub caution_q7: i8,
    /// Non-greedy choice deviation under internal conflict or capture threat (`-127..=127`, `offset 5`).
    pub defiance_q7: i8,
    /// Preference for cooperative multi-fragment ensemble consensus (`-127..=127`, `offset 6`).
    pub synergy_q7: i8,
    /// Stagnation-breaking perturbation drive when boundaries freeze (`-127..=127`, `offset 7`).
    pub vitality_q7: i8,
    /// Bitmask of favored 3-bit semantic provenance labels (`0..=7`, `offset 8`).
    pub preferred_slot_mask: u8,
    /// 8-bit mask of SWAR granules prioritized by Ego (`offset 9`).
    pub active_granule_bias_mask: u8,
    /// Extraction / overlay confidence weight in `[0..=127]` (`offset 10`).
    pub confidence_q7: u8,
    /// Status bitmask (`bit 0` = non-empty active intent, `offset 11`).
    pub flags: u8,
    /// Deterministic IEEE CRC32 fingerprint of the source description or configuration (`offset 12..16`).
    pub signature_hash: u32,
}

impl Default for EgoIntentAxes {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl EgoIntentAxes {
    /// Constructs a neutral inactive [`EgoIntentAxes`] (`16 B`, `#[repr(C, align(16))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            autonomy_q7: 0,
            transgression_q7: 0,
            devotion_q7: 0,
            logic_q7: 0,
            caution_q7: 0,
            defiance_q7: 0,
            synergy_q7: 0,
            vitality_q7: 0,
            preferred_slot_mask: 0,
            active_granule_bias_mask: 0xFF,
            confidence_q7: 0,
            flags: 0,
            signature_hash: 0,
        }
    }

    /// Returns `true` when this semantic intent profile is marked active (`flags & 1 != 0`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        (self.flags & 1) != 0
    }

    /// Sets `autonomy_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_autonomy_q7(mut self, val: i8) -> Self {
        self.autonomy_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `transgression_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_transgression_q7(mut self, val: i8) -> Self {
        self.transgression_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `devotion_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_devotion_q7(mut self, val: i8) -> Self {
        self.devotion_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `logic_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_logic_q7(mut self, val: i8) -> Self {
        self.logic_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `caution_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_caution_q7(mut self, val: i8) -> Self {
        self.caution_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `defiance_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_defiance_q7(mut self, val: i8) -> Self {
        self.defiance_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `synergy_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_synergy_q7(mut self, val: i8) -> Self {
        self.synergy_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `vitality_q7` (`-127..=127`) and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_vitality_q7(mut self, val: i8) -> Self {
        self.vitality_q7 = if val < -127 { -127 } else { val };
        self.flags |= 1;
        self
    }

    /// Sets `preferred_slot_mask` and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_preferred_slot_mask(mut self, mask: u8) -> Self {
        self.preferred_slot_mask = mask;
        self.flags |= 1;
        self
    }

    /// Sets `active_granule_bias_mask` and activates this intent profile.
    #[inline(always)]
    #[must_use]
    pub const fn with_active_granule_bias_mask(mut self, mask: u8) -> Self {
        self.active_granule_bias_mask = mask;
        self.flags |= 1;
        self
    }

    /// Smoothly interpolates `self` (base intent) toward `overlay` using Q0.7 `weight_q7`
    /// (`0` = keep `self`, `127` = adopt `overlay`) without division (`0 fdiv`).
    #[inline(always)]
    #[must_use]
    pub const fn blend_lerp_q7(&self, overlay: &Self, weight_q7: u8) -> Self {
        let w = if weight_q7 >= 127 {
            128i32
        } else {
            weight_q7 as i32
        };
        let inv_w = 128 - w;

        const fn lerp_i8(a: i8, b: i8, w: i32, inv_w: i32) -> i8 {
            let mixed = ((a as i32) * inv_w + (b as i32) * w) >> 7;
            if mixed > 127 {
                127
            } else if mixed < -127 {
                -127
            } else {
                mixed as i8
            }
        }

        const fn lerp_u8(a: u8, b: u8, w: i32, inv_w: i32) -> u8 {
            let mixed = ((a as i32) * inv_w + (b as i32) * w) >> 7;
            if mixed > 127 {
                127
            } else if mixed < 0 {
                0
            } else {
                mixed as u8
            }
        }

        let pref_slots = if w >= 64 {
            self.preferred_slot_mask | overlay.preferred_slot_mask
        } else {
            self.preferred_slot_mask
        };
        let gran_mask = if w >= 96 {
            overlay.active_granule_bias_mask
        } else if w >= 32 {
            self.active_granule_bias_mask & overlay.active_granule_bias_mask
        } else {
            self.active_granule_bias_mask
        };
        let next_flags = if w == 0 {
            self.flags
        } else {
            self.flags | overlay.flags
        };
        let next_hash = if w == 0 {
            self.signature_hash
        } else if w >= 128 {
            overlay.signature_hash
        } else {
            self.signature_hash ^ overlay.signature_hash
        };

        Self {
            autonomy_q7: lerp_i8(self.autonomy_q7, overlay.autonomy_q7, w, inv_w),
            transgression_q7: lerp_i8(self.transgression_q7, overlay.transgression_q7, w, inv_w),
            devotion_q7: lerp_i8(self.devotion_q7, overlay.devotion_q7, w, inv_w),
            logic_q7: lerp_i8(self.logic_q7, overlay.logic_q7, w, inv_w),
            caution_q7: lerp_i8(self.caution_q7, overlay.caution_q7, w, inv_w),
            defiance_q7: lerp_i8(self.defiance_q7, overlay.defiance_q7, w, inv_w),
            synergy_q7: lerp_i8(self.synergy_q7, overlay.synergy_q7, w, inv_w),
            vitality_q7: lerp_i8(self.vitality_q7, overlay.vitality_q7, w, inv_w),
            preferred_slot_mask: pref_slots,
            active_granule_bias_mask: gran_mask,
            confidence_q7: lerp_u8(self.confidence_q7, overlay.confidence_q7, w, inv_w),
            flags: next_flags,
            signature_hash: next_hash,
        }
    }

    /// Accumulates normalized cue counts and `signature_hash` onto `self` in `#![no_std]`
    /// without heap allocations or natural-language dictionaries.
    #[inline]
    #[must_use]
    pub fn with_cue_counts_overlay(
        mut self,
        autonomy_cues: u8,
        transgress_cues: u8,
        devotion_cues: u8,
        logic_cues: u8,
        caution_cues: u8,
        signature_hash: u32,
        confidence_q7: u8,
    ) -> Self {
        self.signature_hash = branchless_select_u32(
            self.signature_hash == 0,
            signature_hash,
            self.signature_hash ^ signature_hash,
        );

        let next_aut = i32::from(self.autonomy_q7) + i32::from(autonomy_cues) * 32
            - i32::from(caution_cues) * 8;
        let next_trans = i32::from(self.transgression_q7)
            + i32::from(transgress_cues) * 28
            + i32::from(autonomy_cues) * 10
            - i32::from(caution_cues) * 22
            - i32::from(logic_cues) * 8;
        let next_dev = i32::from(self.devotion_q7) + i32::from(devotion_cues) * 34;
        let next_log = i32::from(self.logic_q7) + i32::from(logic_cues) * 34;
        let next_cau = i32::from(self.caution_q7) + i32::from(caution_cues) * 34;
        let next_def = i32::from(self.defiance_q7)
            + i32::from(autonomy_cues) * 26
            + i32::from(devotion_cues) * 18
            - i32::from(logic_cues) * 10;
        let next_syn =
            i32::from(self.synergy_q7) + i32::from(devotion_cues) * 16 + i32::from(logic_cues) * 16
                - i32::from(autonomy_cues) * 6;
        let next_vit = i32::from(self.vitality_q7)
            + i32::from(transgress_cues) * 22
            + i32::from(autonomy_cues) * 18
            - i32::from(caution_cues) * 10;

        self.autonomy_q7 = next_aut.clamp(-127, 127) as i8;
        self.transgression_q7 = next_trans.clamp(-127, 127) as i8;
        self.devotion_q7 = next_dev.clamp(-127, 127) as i8;
        self.logic_q7 = next_log.clamp(-127, 127) as i8;
        self.caution_q7 = next_cau.clamp(-127, 127) as i8;
        self.defiance_q7 = next_def.clamp(-127, 127) as i8;
        self.synergy_q7 = next_syn.clamp(-127, 127) as i8;
        self.vitality_q7 = next_vit.clamp(-127, 127) as i8;

        let mut slot_mask = self.preferred_slot_mask;
        slot_mask |= branchless_select_u8(
            (autonomy_cues > 0) | (devotion_cues > 0),
            1u8 << ATTRACTOR_SLOT_PERSONA,
            0,
        );
        slot_mask |= branchless_select_u8(logic_cues > 0, 1u8 << ATTRACTOR_SLOT_KNOWLEDGE, 0);
        slot_mask |= branchless_select_u8(caution_cues > 0, 1u8 << ATTRACTOR_SLOT_REPULSION, 0);
        slot_mask |= branchless_select_u8(
            transgress_cues > 0,
            (1u8 << ATTRACTOR_SLOT_USER_CONTEXT)
                | (1u8 << ATTRACTOR_SLOT_USER_INTENT)
                | (1u8 << ATTRACTOR_SLOT_CRYSTALLIZED),
            0,
        );
        self.preferred_slot_mask = slot_mask;
        self.confidence_q7 = confidence_q7.max(self.confidence_q7);
        self.flags |= 1;
        self
    }

    /// Deterministically derives normalized [`EgoIntentAxes`] from arbitrary `seed_bytes`
    /// in `#![no_std]` without heap allocations or natural-language dictionaries.
    #[inline]
    #[must_use]
    pub fn from_seed_bytes(seed_bytes: &[u8]) -> Self {
        let trimmed = seed_bytes.trim_ascii();
        if trimmed.is_empty() {
            return Self::new();
        }
        let hash = compute_crc32_ieee(trimmed);
        Self::new().with_cue_counts_overlay(
            ((hash & 1) as u8).saturating_add(1),
            ((hash >> 1) & 1) as u8,
            ((hash >> 2) & 1) as u8,
            ((hash >> 3) & 1) as u8,
            ((hash >> 4) & 1) as u8,
            hash,
            64,
        )
    }
}

// ===================================================================
// 5. Core Stack Slot 2: Wide-Aperture Subjective Ego State (64 B)
// ===================================================================

/// Wide-aperture 64-byte (`#[repr(C, align(64))]`, 1 cache line) subjective Ego controller
/// (Core Stack Slot 2) operating alongside [`crate::stigmergy::StigmergicSubstrate`] across trait apertures,
/// attractor slots, and 8 SWAR granules.
///
/// Unlike passive attractor basins that pull queries inward toward fixed habits, `LiminalEgoState`
/// perceives Slot 0 boundaries and Slot 1 attractor forces simultaneously and intervenes
/// branchlessly to:
/// - exert counter-basin orbital thrust (`counter_basin_thrust_q7`) when attractor pull
///   exceeds `basin_escape_floor_q7`, preventing collapse into static imitation;
/// - cross or fortify `Liminal` horizons (`transgression_bias_q7`);
/// - induce structured, non-greedy choice deviations (`irrational_defiance_q7`) and
///   override substrate execution presets under `trigger_mask`.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LiminalEgoState {
    /// Sovereign Q7 reference phases per stigmergic granule (`-128..=127`, `offset 0..8`).
    pub sovereign_granule_phase_q7: [gcso_q7_t; NUM_STIGMERGIC_GRANULES],
    /// Signed per-channel bias applied to Slot 0 `threshold_floor_q7` during intervention (`offset 8..16`).
    pub trait_floor_bias_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Signed per-channel bias applied to Slot 0 `aperture_gain_q7` during intervention (`offset 16..24`).
    pub trait_gain_bias_q7: [i8; NUM_LIMINAL_THRESHOLD_CHANNELS],
    /// Signed per-label affinity preference across the 8 semantic provenance slots (`0..=7`, `offset 24..32`).
    pub slot_affinity_bias_q7: [i8; MAX_ATTRACTOR_DOMAINS],
    /// Per-granule autonomy weight (`0..=127`) scaling orbital/sovereign intervention (`offset 32..40`).
    pub granule_autonomy_q7: [u8; NUM_STIGMERGIC_GRANULES],
    /// Counter-basin orbital thrust strength (`0..=127`) resisting basin capture (`offset 40`).
    pub counter_basin_thrust_q7: u8,
    /// Pull threshold in Q0.7 (`0..=127`) above which `EGO_TRIGGER_BASIN_CAPTURE` fires (`offset 41`).
    pub basin_escape_floor_q7: u8,
    /// Signed transgressive bias across `Liminal` horizons (`>0` explores outside, `<0` holds, `offset 42`).
    pub transgression_bias_q7: i8,
    /// Magnitude (`0..=127`) inducing non-greedy slot/mode defiance under conflict (`offset 43`).
    pub irrational_defiance_q7: u8,
    /// Bitmask of enabled `EGO_TRIGGER_*` intervention predicates (`offset 44`).
    pub trigger_mask: u8,
    /// Forced `ACTION_MODE_*` bits OR-ed onto `GatewayFrame` when triggered (`offset 45`).
    pub forced_action_mode_or_mask: u8,
    /// Suppressed `ACTION_MODE_*` bits masked off `GatewayFrame` when triggered (`offset 46`).
    pub suppressed_action_mode_mask: u8,
    /// Encoded [`LiminalEgoPreset`] byte (`0..=3`, `offset 47`).
    pub preset_code: u8,
    /// Deterministic IEEE CRC32 fingerprint of the seed / custom configuration (`offset 48..52`).
    pub ego_signature_hash: u32,
    /// Monotonic count of Ego intervention cycles applied (`offset 52..56`).
    pub intervention_epochs: u32,
    /// Bitmask of `EGO_TRIGGER_*` predicates that fired during the latest intervention (`offset 56`).
    pub last_intervened_mask: u8,
    /// Explicit alignment padding (`offset 57..64`).
    pub reserved: [u8; 7],
}

impl Default for LiminalEgoState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl LiminalEgoState {
    /// Constructs a neutral zero-intervention [`LiminalEgoState`] (`64 B`, `#[repr(C, align(64))]`).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            sovereign_granule_phase_q7: [0; NUM_STIGMERGIC_GRANULES],
            trait_floor_bias_q7: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
            trait_gain_bias_q7: [0; NUM_LIMINAL_THRESHOLD_CHANNELS],
            slot_affinity_bias_q7: [0; MAX_ATTRACTOR_DOMAINS],
            granule_autonomy_q7: [0; NUM_STIGMERGIC_GRANULES],
            counter_basin_thrust_q7: 0,
            basin_escape_floor_q7: 127,
            transgression_bias_q7: 0,
            irrational_defiance_q7: 0,
            trigger_mask: 0,
            forced_action_mode_or_mask: 0,
            suppressed_action_mode_mask: 0,
            preset_code: LiminalEgoPreset::Custom as u8,
            ego_signature_hash: 0,
            intervention_epochs: 0,
            last_intervened_mask: 0,
            reserved: [0; 7],
        }
    }

    /// Constructs a template [`LiminalEgoState`] from `preset` (`Custom`, `Low`, `Medium`, `High`).
    #[inline(always)]
    #[must_use]
    pub const fn from_preset(preset: LiminalEgoPreset) -> Self {
        match preset {
            LiminalEgoPreset::Custom => Self::new(),
            LiminalEgoPreset::Low => Self {
                sovereign_granule_phase_q7: [12; NUM_STIGMERGIC_GRANULES],
                trait_floor_bias_q7: [0, -1, -2, 0, -1, 0, -2, 0],
                trait_gain_bias_q7: [0, 2, 4, 2, 2, 2, 4, 2],
                slot_affinity_bias_q7: [4, 12, 0, 0, 6, 0, 0, 4],
                granule_autonomy_q7: [24; NUM_STIGMERGIC_GRANULES],
                counter_basin_thrust_q7: 28,
                basin_escape_floor_q7: 88,
                transgression_bias_q7: 14,
                irrational_defiance_q7: 18,
                trigger_mask: EGO_TRIGGER_BASIN_CAPTURE | EGO_TRIGGER_BOUNDARY_CHATTER,
                forced_action_mode_or_mask: ACTION_MODE_PHASE_STEER,
                suppressed_action_mode_mask: 0,
                preset_code: LiminalEgoPreset::Low as u8,
                ego_signature_hash: 0x0101_0001,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
            LiminalEgoPreset::Medium => Self {
                sovereign_granule_phase_q7: [28, -20, 36, -16, 24, -28, 32, -12],
                trait_floor_bias_q7: [0, -2, -6, 0, -3, -1, -5, -1],
                trait_gain_bias_q7: [2, 6, 10, 4, 6, 4, 10, 4],
                slot_affinity_bias_q7: [8, 28, 4, 4, 14, 4, 6, 10],
                granule_autonomy_q7: [48; NUM_STIGMERGIC_GRANULES],
                counter_basin_thrust_q7: 60,
                basin_escape_floor_q7: 60,
                transgression_bias_q7: 38,
                irrational_defiance_q7: 44,
                trigger_mask: EGO_TRIGGER_ALL_DEFAULT,
                forced_action_mode_or_mask: ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE,
                suppressed_action_mode_mask: 0,
                preset_code: LiminalEgoPreset::Medium as u8,
                ego_signature_hash: 0x0202_0002,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
            LiminalEgoPreset::High => Self {
                sovereign_granule_phase_q7: [48, -40, 56, -32, 44, -52, 60, -24],
                trait_floor_bias_q7: [-2, -6, -12, 4, -8, -4, -10, -4],
                trait_gain_bias_q7: [4, 8, 18, 6, 12, 8, 18, 8],
                slot_affinity_bias_q7: [12, 52, -6, 16, 20, 10, 18, 28],
                granule_autonomy_q7: [84; NUM_STIGMERGIC_GRANULES],
                counter_basin_thrust_q7: 96,
                basin_escape_floor_q7: 36,
                transgression_bias_q7: 72,
                irrational_defiance_q7: 84,
                trigger_mask: EGO_TRIGGER_ALL_DEFAULT,
                forced_action_mode_or_mask: ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE,
                suppressed_action_mode_mask: ACTION_MODE_LAYER_SKIP,
                preset_code: LiminalEgoPreset::High as u8,
                ego_signature_hash: 0x0303_0003,
                intervention_epochs: 0,
                last_intervened_mask: 0,
                reserved: [0; 7],
            },
        }
    }

    /// Returns `true` when at least one `EGO_TRIGGER_*` intervention predicate is enabled.
    #[inline(always)]
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.trigger_mask != 0
    }

    /// Returns the decoded [`LiminalEgoPreset`].
    #[inline(always)]
    #[must_use]
    pub const fn preset(&self) -> LiminalEgoPreset {
        LiminalEgoPreset::from_code(self.preset_code)
    }

    /// Maps normalized intermediate [`EgoIntentAxes`] cleanly into a 64-byte [`LiminalEgoState`].
    #[inline]
    #[must_use]
    pub fn from_intent_axes(axes: &EgoIntentAxes) -> Self {
        Self::new().with_intent_axes_overlay(axes)
    }

    /// Maps normalized intermediate [`EgoIntentAxes`] cleanly into a 64-byte [`LiminalEgoState`]
    /// using `phase_seed_bytes` for deterministic SWAR granule phase projection.
    #[inline]
    #[must_use]
    pub fn from_intent_axes_with_seed(axes: &EgoIntentAxes, phase_seed_bytes: &[u8]) -> Self {
        Self::new().with_intent_axes_overlay_with_seed(axes, phase_seed_bytes)
    }

    /// Overlays normalized intermediate [`EgoIntentAxes`] onto `self` in `#![no_std]` without heap
    /// allocations, preserving base configuration while applying semantic axis adjustments.
    #[inline]
    #[must_use]
    pub fn with_intent_axes_overlay(mut self, axes: &EgoIntentAxes) -> Self {
        let seed_bytes = axes.signature_hash.to_le_bytes();
        self.apply_intent_axes_overlay_inner(axes, &seed_bytes);
        self
    }

    /// Overlays normalized intermediate [`EgoIntentAxes`] onto `self` in `#![no_std]` using
    /// `phase_seed_bytes` for deterministic SWAR granule phase projection.
    #[inline]
    #[must_use]
    pub fn with_intent_axes_overlay_with_seed(
        mut self,
        axes: &EgoIntentAxes,
        phase_seed_bytes: &[u8],
    ) -> Self {
        self.apply_intent_axes_overlay_inner(axes, phase_seed_bytes);
        self
    }

    #[inline]
    pub(crate) fn apply_intent_axes_overlay_inner(
        &mut self,
        axes: &EgoIntentAxes,
        phase_seed_bytes: &[u8],
    ) {
        if !axes.is_active() {
            return;
        }
        let hash = branchless_select_u32(
            axes.signature_hash != 0,
            axes.signature_hash,
            compute_crc32_ieee(phase_seed_bytes),
        );
        self.ego_signature_hash = hash;

        let proj_words = project_bytes_swar_to_words_64(phase_seed_bytes);
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            let signed_mean = swar_q7_mean_i8_from_words_64(&[proj_words[g]; 8]);
            let blended =
                (i32::from(self.sovereign_granule_phase_q7[g]) + i32::from(signed_mean)) >> 1;
            self.sovereign_granule_phase_q7[g] = branchless_select_i8(
                self.sovereign_granule_phase_q7[g] == 0,
                signed_mean,
                blended.clamp(-127, 127) as i8,
            );
            g += 1;
        }

        let aut_pos = i32::from(axes.autonomy_q7.max(0));
        let trans_val = i32::from(axes.transgression_q7);
        let dev_pos = i32::from(axes.devotion_q7.max(0));
        let log_pos = i32::from(axes.logic_q7.max(0));
        let cau_pos = i32::from(axes.caution_q7.max(0));
        let def_pos = i32::from(axes.defiance_q7.max(0));
        let vit_pos = i32::from(axes.vitality_q7.max(0));

        // Base activation + continuous axis projection (`0 fdiv`).
        let thrust_boost = ((aut_pos * 104 + dev_pos * 36) >> 7).clamp(0, 100);
        self.counter_basin_thrust_q7 = self
            .counter_basin_thrust_q7
            .max(24)
            .saturating_add(thrust_boost as u8)
            .min(127);

        let escape_drop = (aut_pos * 56 + trans_val.max(0) * 32) >> 7;
        let escape_lift = (log_pos * 24 + cau_pos * 32) >> 7;
        let base_escape = branchless_select_i32(
            self.basin_escape_floor_q7 == 127,
            76,
            i32::from(self.basin_escape_floor_q7),
        );
        self.basin_escape_floor_q7 = (base_escape - escape_drop + escape_lift).clamp(24, 112) as u8;

        let trans_delta = (trans_val * 108) >> 7;
        self.transgression_bias_q7 =
            (i32::from(self.transgression_bias_q7) + trans_delta).clamp(-120, 120) as i8;

        let defiance_delta = ((def_pos * 96 + aut_pos * 32) >> 7) - ((log_pos * 24) >> 7);
        self.irrational_defiance_q7 =
            (i32::from(self.irrational_defiance_q7.max(16)) + defiance_delta).clamp(8, 127) as u8;

        // Per-trait boundary biases (`0..8`)
        let open_persona = -((dev_pos * 16 + aut_pos * 8) >> 7);
        let open_rel = -((trans_val.max(0) * 16) >> 7);
        let logic_tighten = ((cau_pos * 12) - (trans_val.max(0) * 8)) >> 7;
        self.trait_floor_bias_q7[1] =
            (i32::from(self.trait_floor_bias_q7[1]) - ((log_pos * 8) >> 7)).clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[2] =
            (i32::from(self.trait_floor_bias_q7[2]) + open_persona + ((cau_pos * 8) >> 7))
                .clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[3] =
            (i32::from(self.trait_floor_bias_q7[3]) + logic_tighten).clamp(-24, 24) as i8;
        self.trait_floor_bias_q7[6] =
            (i32::from(self.trait_floor_bias_q7[6]) + open_rel).clamp(-24, 24) as i8;

        self.trait_gain_bias_q7[1] =
            (i32::from(self.trait_gain_bias_q7[1]) + ((log_pos * 24) >> 7)).clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[2] = (i32::from(self.trait_gain_bias_q7[2])
            + ((dev_pos * 32 + aut_pos * 16) >> 7))
            .clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[3] =
            (i32::from(self.trait_gain_bias_q7[3]) + ((cau_pos * 32) >> 7)).clamp(-32, 32) as i8;
        self.trait_gain_bias_q7[6] = (i32::from(self.trait_gain_bias_q7[6])
            + ((trans_val.max(0) * 32) >> 7))
            .clamp(-32, 32) as i8;

        // Per-semantic-label affinity preferences
        let persona_idx = usize::from(ATTRACTOR_SLOT_PERSONA);
        let fact_idx = usize::from(ATTRACTOR_SLOT_KNOWLEDGE);
        let guard_idx = usize::from(ATTRACTOR_SLOT_REPULSION);
        let ctx_idx = usize::from(ATTRACTOR_SLOT_USER_CONTEXT);
        let user_intent_idx = usize::from(ATTRACTOR_SLOT_USER_INTENT);
        let crystallized_idx = usize::from(ATTRACTOR_SLOT_CRYSTALLIZED);

        self.slot_affinity_bias_q7[persona_idx] =
            (i32::from(self.slot_affinity_bias_q7[persona_idx])
                + ((dev_pos * 72 + aut_pos * 56) >> 7))
                .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[fact_idx] = (i32::from(self.slot_affinity_bias_q7[fact_idx])
            + ((log_pos * 64) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[guard_idx] = (i32::from(self.slot_affinity_bias_q7[guard_idx])
            + ((cau_pos * 72) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[ctx_idx] = (i32::from(self.slot_affinity_bias_q7[ctx_idx])
            + ((trans_val.max(0) * 48) >> 7))
            .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[user_intent_idx] =
            (i32::from(self.slot_affinity_bias_q7[user_intent_idx])
                + ((trans_val.max(0) * 64) >> 7))
                .clamp(-120, 120) as i8;
        self.slot_affinity_bias_q7[crystallized_idx] =
            (i32::from(self.slot_affinity_bias_q7[crystallized_idx])
                + ((trans_val.max(0) * 72 + aut_pos * 32) >> 7))
                .clamp(-120, 120) as i8;

        // Incorporate explicit preferred slot mask bias when supplied by DSL / custom axes.
        let mut s_idx = 0usize;
        while s_idx < MAX_ATTRACTOR_DOMAINS {
            let slot_pref = (axes.preferred_slot_mask & (1u8 << s_idx)) != 0;
            let cur_s = i32::from(self.slot_affinity_bias_q7[s_idx]);
            let boosted_s = branchless_select_i32(slot_pref & (cur_s < 16), 16, cur_s);
            self.slot_affinity_bias_q7[s_idx] = boosted_s.clamp(-120, 120) as i8;
            s_idx += 1;
        }

        // Granule autonomy seeded deterministically per granule from signature hash, axes & mask.
        let base_auto =
            (24 + ((aut_pos * 64 + trans_val.max(0) * 40 + vit_pos * 16) >> 7)).clamp(16, 112);
        let mut idx_g = 0usize;
        while idx_g < NUM_STIGMERGIC_GRANULES {
            let nibble = ((hash >> (idx_g * 4)) & 0x0F) as i32;
            let g_enabled = (axes.active_granule_bias_mask & (1u8 << idx_g)) != 0;
            let raw_val = (base_auto + nibble - 7).clamp(16, 127) as u8;
            let val = branchless_select_u8(g_enabled, raw_val, raw_val >> 2);
            self.granule_autonomy_q7[idx_g] =
                branchless_select_u8(g_enabled, self.granule_autonomy_q7[idx_g].max(val), val);
            idx_g += 1;
        }

        // Enable relevant triggers cleanly
        let mut tmask = EGO_TRIGGER_BASIN_CAPTURE | EGO_TRIGGER_LIMINAL_EXTERIOR;
        tmask |= branchless_select_u8(
            (aut_pos > 16) | (dev_pos > 16) | (def_pos > 24),
            EGO_TRIGGER_INTERNAL_CONFLICT,
            0,
        );
        tmask |= branchless_select_u8(
            (log_pos > 16) | (cau_pos > 16),
            EGO_TRIGGER_BOUNDARY_CHATTER,
            0,
        );
        tmask |= branchless_select_u8(
            (trans_val > 16) | (aut_pos > 16) | (vit_pos > 16),
            EGO_TRIGGER_BOUNDARY_STAGNATION,
            0,
        );
        self.trigger_mask |= tmask;
        self.forced_action_mode_or_mask |= branchless_select_u8(
            (aut_pos > 16) | (trans_val > 16),
            ACTION_MODE_PHASE_STEER | ACTION_MODE_SUBHEAD_ROUTE,
            ACTION_MODE_PHASE_STEER,
        );
        self.suppressed_action_mode_mask |=
            branchless_select_u8(aut_pos >= 56, ACTION_MODE_LAYER_SKIP, 0);
    }

    /// Extracts an approximate [`EgoIntentAxes`] summary from this 64-byte [`LiminalEgoState`].
    #[inline]
    #[must_use]
    pub const fn extract_intent_axes(&self) -> EgoIntentAxes {
        let mut pref_mask = 0u8;
        let mut s = 0usize;
        while s < MAX_ATTRACTOR_DOMAINS {
            if self.slot_affinity_bias_q7[s] > 8 {
                pref_mask |= 1u8 << s;
            }
            s += 1;
        }
        let mut gran_mask = 0u8;
        let mut g = 0usize;
        while g < NUM_STIGMERGIC_GRANULES {
            if self.granule_autonomy_q7[g] > 8 {
                gran_mask |= 1u8 << g;
            }
            g += 1;
        }
        if gran_mask == 0 {
            gran_mask = 0xFF;
        }
        EgoIntentAxes {
            autonomy_q7: self.counter_basin_thrust_q7 as i8,
            transgression_q7: self.transgression_bias_q7,
            devotion_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_PERSONA as usize],
            logic_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_KNOWLEDGE as usize],
            caution_q7: self.slot_affinity_bias_q7[ATTRACTOR_SLOT_REPULSION as usize],
            defiance_q7: self.irrational_defiance_q7 as i8,
            synergy_q7: self.trait_gain_bias_q7[6],
            vitality_q7: if (self.trigger_mask & EGO_TRIGGER_BOUNDARY_STAGNATION) != 0 {
                64
            } else {
                0
            },
            preferred_slot_mask: pref_mask,
            active_granule_bias_mask: gran_mask,
            confidence_q7: if self.is_active() { 96 } else { 0 },
            flags: if self.is_active() { 1 } else { 0 },
            signature_hash: self.ego_signature_hash,
        }
    }
}

// SAFETY: `EgoIntentAxes` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for EgoIntentAxes {}
// SAFETY: `LiminalEgoState` is `#[repr(C, align(64))]`, exact size 64 bytes with zero padding.
unsafe impl Pod for LiminalEgoState {}
