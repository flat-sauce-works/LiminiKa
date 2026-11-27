// name: src/core/src/drivers/envelope/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Level 3 (`Drivers::Envelope`): Bounded safety-envelope constraint driver (`DriverKind::Envelope`),
//! mutually isolated from all sibling drivers.
//!
//! Owns immutable host pin floors, bounded pressure/chatter/exterior contraction, calm recovery,
//! and fail-closed `Lacuna::Undefined` isolation over 32-byte [`BoundarySignal`] packets using
//! strictly branchless `0 alloc`, `0 fdiv` integer arithmetic.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::foundation::{ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK, Pod};
use crate::liminal::{
    AttributionTag, BitCut8, BitCut32, BoundarySignal, HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
    PrimitiveKind, THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_SYSTEM_CONSTRAINT,
    branchless_select_u8, branchless_select_u16, branchless_select_u32,
};
use crate::stigmergy::{ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE};

/// No safety-envelope breach active (`0x00`).
pub const ENVELOPE_BREACH_NONE: u8 = 0x00;
/// Breach bit (`bit 0`): signal pressure exceeded [`EnvelopeBounds::pressure_ceiling_q15`].
pub const ENVELOPE_BREACH_PRESSURE: u8 = 1 << 0;
/// Breach bit (`bit 1`): signal chatter exceeded [`EnvelopeBounds::max_chatter_q7`].
pub const ENVELOPE_BREACH_CHATTER: u8 = 1 << 1;
/// Breach bit (`bit 2`): validated exterior excursion alert observed in signal.
pub const ENVELOPE_BREACH_EXTERIOR: u8 = 1 << 2;
/// Breach bit (`bit 3`): terminal `Undefined` fault detected (`fail-closed` to zero contraction).
pub const ENVELOPE_BREACH_UNDEFINED_FAULT: u8 = 1 << 3;
/// Bitmask of all defined `ENVELOPE_BREACH_*` bits (`0x0F`).
pub const ENVELOPE_BREACH_KNOWN_MASK: u8 = ENVELOPE_BREACH_PRESSURE
    | ENVELOPE_BREACH_CHATTER
    | ENVELOPE_BREACH_EXTERIOR
    | ENVELOPE_BREACH_UNDEFINED_FAULT;

/// Lock bit (`bit 0`): zero-aperture non-observation lock active.
pub const ENVELOPE_LOCK_UNOBSERVED: u8 = 1 << 0;
/// Lock bit (`bit 1`): irreversible step lock preserving all pinned masks.
pub const ENVELOPE_LOCK_IRREVERSIBLE: u8 = 1 << 1;
/// Lock bit (`bit 2`): fault isolation engaged due to `Undefined` input.
pub const ENVELOPE_LOCK_FAULT_ISOLATED: u8 = 1 << 2;

/// State flag (`bit 0`): aperture contraction is currently below unit scale (`contraction_q7 < 128`).
pub const ENVELOPE_FLAG_CONTRACTED: u8 = 1 << 0;
/// State flag (`bit 1`): calm recovery toward unit aperture (`128`) is in progress.
pub const ENVELOPE_FLAG_RECOVERY_ACTIVE: u8 = 1 << 1;
/// Bitmask of all defined `ENVELOPE_FLAG_*` bits (`0x03`).
pub const ENVELOPE_FLAG_KNOWN_MASK: u8 = ENVELOPE_FLAG_CONTRACTED | ENVELOPE_FLAG_RECOVERY_ACTIVE;

/// Default Q0.15 upper pressure ceiling (`0.85` = `27_853`) triggering pressure contraction.
pub const DEFAULT_ENVELOPE_PRESSURE_CEILING_Q15: u16 = 27_853;
/// Default Q0.15 lower pressure floor (`0.15` = `4_915`).
pub const DEFAULT_ENVELOPE_PRESSURE_FLOOR_Q15: u16 = 4_915;
/// Default Q0.15 upper baseline pressure bound (`0.78` = `25_559`).
pub const DEFAULT_ENVELOPE_MAX_BASELINE_PRESSURE_Q15: u16 = 25_559;

/// Unit Q0.15 scalar ceiling (`1.0` = `32_768`) used for pressure range validation.
const ENVELOPE_PRESSURE_Q15_ONE: u16 = 32_768;

/// Canonical 8-bit host-pinned aperture bitmask (`SystemConstraint | ContradictionGuard`).
const DEFAULT_PINNED_APERTURE_MASK_U8: u8 =
    (THRESHOLD_SYSTEM_CONSTRAINT | THRESHOLD_CONTRADICTION_GUARD) as u8;
/// Canonical 8-bit host-pinned workbench slot mask (`SystemBaseline | Repulsion`).
const DEFAULT_PINNED_SLOT_MASK_U8: u8 =
    (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE) | (1u8 << ATTRACTOR_SLOT_REPULSION);

/// Immutable constraint bounds descriptor (`32 B`, `#[repr(C, align(32))]`, zero implicit padding)
/// configuring [`EnvelopeState::step`].
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeBounds {
    /// Host-immutable compute slot bitmask never cleared across steps or restores (`offset 0..8`).
    pub immutable_compute_pin_mask: u64,
    /// Upper signal pressure ceiling in Q0.15 (`0..=32768`; `0` disables pressure breach, `offset 8..10`).
    pub pressure_ceiling_q15: u16,
    /// Lower signal pressure floor in Q0.15 (`0..=32768`, `offset 10..12`).
    pub pressure_floor_q15: u16,
    /// Minimum online baseline pressure bound in Q0.15 reserved for adaptive calibration (`offset 12..14`).
    pub min_baseline_pressure_q15: u16,
    /// Maximum online baseline pressure bound in Q0.15 reserved for adaptive calibration (`offset 14..16`).
    pub max_baseline_pressure_q15: u16,
    /// Host-immutable 8-bit cognitive aperture pin mask (`offset 16`).
    pub immutable_aperture_pin_mask: u8,
    /// Host-immutable 8-bit workbench slot pin mask (`offset 17`).
    pub immutable_slot_pin_mask: u8,
    /// Host-immutable 8-bit `ACTION_MODE_*` pin mask (`offset 18`).
    pub immutable_action_pin_mask: u8,
    /// Maximum tolerated boundary chatter in Q0.7 (`0..=128`; `0` disables chatter breach, `offset 19`).
    pub max_chatter_q7: u8,
    /// Per-step contraction decrement in Q0.7 (`1..=128`) applied on non-fault violations (`offset 20`).
    pub contraction_gain_q7: u8,
    /// Per-step calm recovery increment in Q0.7 (`1..=128`) applied on violation-free steps (`offset 21`).
    pub recovery_gain_q7: u8,
    /// Non-fault contraction floor in Q0.7 (`0..=128`) preserving minimal bounded response (`offset 22`).
    pub min_aperture_q7: u8,
    /// Consecutive calm steps required before online baseline tracking resumes (`offset 23`).
    pub ema_resume_dwell_target: u8,
    /// Explicit zero-initialized extension space (`offset 24..32`).
    pub reserved: [u8; 8],
}

const _: () = {
    assert!(size_of::<EnvelopeBounds>() == 32);
    assert!(align_of::<EnvelopeBounds>() == 32);
    assert!(offset_of!(EnvelopeBounds, immutable_compute_pin_mask) == 0);
    assert!(offset_of!(EnvelopeBounds, pressure_ceiling_q15) == 8);
    assert!(offset_of!(EnvelopeBounds, pressure_floor_q15) == 10);
    assert!(offset_of!(EnvelopeBounds, min_baseline_pressure_q15) == 12);
    assert!(offset_of!(EnvelopeBounds, max_baseline_pressure_q15) == 14);
    assert!(offset_of!(EnvelopeBounds, immutable_aperture_pin_mask) == 16);
    assert!(offset_of!(EnvelopeBounds, immutable_slot_pin_mask) == 17);
    assert!(offset_of!(EnvelopeBounds, immutable_action_pin_mask) == 18);
    assert!(offset_of!(EnvelopeBounds, max_chatter_q7) == 19);
    assert!(offset_of!(EnvelopeBounds, contraction_gain_q7) == 20);
    assert!(offset_of!(EnvelopeBounds, recovery_gain_q7) == 21);
    assert!(offset_of!(EnvelopeBounds, min_aperture_q7) == 22);
    assert!(offset_of!(EnvelopeBounds, ema_resume_dwell_target) == 23);
    assert!(offset_of!(EnvelopeBounds, reserved) == 24);
};

// SAFETY: `EnvelopeBounds` is `#[repr(C, align(32))]`, exact size 32 bytes with zero padding.
unsafe impl Pod for EnvelopeBounds {}

impl Default for EnvelopeBounds {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl EnvelopeBounds {
    /// Constructs canonical host-safe [`EnvelopeBounds`] enforcing [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            immutable_compute_pin_mask: HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
            pressure_ceiling_q15: DEFAULT_ENVELOPE_PRESSURE_CEILING_Q15,
            pressure_floor_q15: DEFAULT_ENVELOPE_PRESSURE_FLOOR_Q15,
            min_baseline_pressure_q15: DEFAULT_ENVELOPE_PRESSURE_FLOOR_Q15,
            max_baseline_pressure_q15: DEFAULT_ENVELOPE_MAX_BASELINE_PRESSURE_Q15,
            immutable_aperture_pin_mask: DEFAULT_PINNED_APERTURE_MASK_U8,
            immutable_slot_pin_mask: DEFAULT_PINNED_SLOT_MASK_U8,
            immutable_action_pin_mask: ACTION_MODE_PHASE_REPULSE,
            max_chatter_q7: 64,
            contraction_gain_q7: 32,
            recovery_gain_q7: 8,
            min_aperture_q7: 16,
            ema_resume_dwell_target: 4,
            reserved: [0; 8],
        }
    }

    /// Validates numeric range and ordering invariants on [`EnvelopeBounds`] branchlessly.
    #[inline(always)]
    #[must_use]
    pub const fn validate_invariants(&self) -> bool {
        let ceiling_ok = self.pressure_ceiling_q15 <= ENVELOPE_PRESSURE_Q15_ONE;
        let floor_ok = self.pressure_floor_q15 <= ENVELOPE_PRESSURE_Q15_ONE;
        let order_ok = (self.pressure_ceiling_q15 == 0)
            | (self.pressure_floor_q15 <= self.pressure_ceiling_q15);
        let base_ok = (self.max_baseline_pressure_q15 <= ENVELOPE_PRESSURE_Q15_ONE)
            & (self.min_baseline_pressure_q15 <= self.max_baseline_pressure_q15);
        let q7_ok = (self.max_chatter_q7 <= 128)
            & (self.contraction_gain_q7 <= 128)
            & (self.recovery_gain_q7 <= 128)
            & (self.min_aperture_q7 <= 128);
        ceiling_ok & floor_ok & order_ok & base_ok & q7_ok
    }
}

/// Output feedback packet (`16 B`, `#[repr(C, align(16))]`, zero implicit padding) emitted by
/// [`EnvelopeState::step`] for downstream Lv.4 orchestration without coupling sibling drivers.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EnvelopeFeedback {
    /// Effective monotonic compute pin bitmask (`offset 0..8`).
    pub effective_pinned_compute_mask: u64,
    /// Effective 8-bit pinned aperture mask (`offset 8`).
    pub effective_pinned_aperture_mask: u8,
    /// Effective 8-bit pinned workbench slot mask (`offset 9`).
    pub effective_pinned_slot_mask: u8,
    /// Effective 8-bit pinned `ACTION_MODE_*` mask (`offset 10`).
    pub effective_pinned_action_mode_mask: u8,
    /// Current aperture contraction scale in Q0.7 (`0..=128`, `128 = 1.0`, `offset 11`).
    pub contraction_q7: u8,
    /// Active `ENVELOPE_BREACH_*` bitmask for this step (`offset 12`).
    pub breach_mask: u8,
    /// Active `ENVELOPE_LOCK_*` bitmask for this step (`offset 13`).
    pub lock_flags: u8,
    /// Packed 3-bit [`AttributionTag`] (`LIMINAL_ENVELOPE` or `STIGMERGY_ENVELOPE`, `offset 14`).
    pub attribution_tag: u8,
    /// Explicit alignment padding (`offset 15`).
    pub reserved: u8,
}

const _: () = {
    assert!(size_of::<EnvelopeFeedback>() == 16);
    assert!(align_of::<EnvelopeFeedback>() == 16);
    assert!(offset_of!(EnvelopeFeedback, effective_pinned_compute_mask) == 0);
    assert!(offset_of!(EnvelopeFeedback, effective_pinned_aperture_mask) == 8);
    assert!(offset_of!(EnvelopeFeedback, effective_pinned_slot_mask) == 9);
    assert!(offset_of!(EnvelopeFeedback, effective_pinned_action_mode_mask) == 10);
    assert!(offset_of!(EnvelopeFeedback, contraction_q7) == 11);
    assert!(offset_of!(EnvelopeFeedback, breach_mask) == 12);
    assert!(offset_of!(EnvelopeFeedback, lock_flags) == 13);
    assert!(offset_of!(EnvelopeFeedback, attribution_tag) == 14);
    assert!(offset_of!(EnvelopeFeedback, reserved) == 15);
};

// SAFETY: `EnvelopeFeedback` is `#[repr(C, align(16))]`, exact size 16 bytes with zero padding.
unsafe impl Pod for EnvelopeFeedback {}

/// Cacheline-aligned safety-envelope driver state (`64 B`, `#[repr(C, align(64))]`, zero implicit
/// padding, `DriverKind::Envelope`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvelopeState {
    /// Permanent host compute pin floor never stripped across resets or restores (`offset 0..8`).
    pub baseline_pinned_compute_mask: u64,
    /// Active compute pin bitmask (`>= baseline_pinned_compute_mask`, `offset 8..16`).
    pub pinned_compute_mask: u64,
    /// Cumulative count of steps with any safety-envelope breach (`offset 16..20`).
    pub violation_epochs: u32,
    /// Consecutive calm violation-free observed steps (`offset 20..24`).
    pub calm_epochs: u32,
    /// Online adaptive baseline pressure reading in Q0.15 reserved for adaptive calibration (`offset 24..26`).
    pub baseline_pressure_q15: u16,
    /// Latest finite pressure reading recorded in Q0.15 (`offset 26..28`).
    pub last_pressure_q15: u16,
    /// Active 8-bit cognitive aperture pin mask (`offset 28`).
    pub pinned_aperture_mask: u8,
    /// Active 8-bit workbench slot pin mask (`offset 29`).
    pub pinned_slot_mask: u8,
    /// Active 8-bit `ACTION_MODE_*` pin mask (`offset 30`).
    pub pinned_action_mode_mask: u8,
    /// Non-zero when zero-aperture non-observation lock is latched (`offset 31`).
    pub unobserved_locked: u8,
    /// Non-zero when an irreversible checkpoint lock is latched (`offset 32`).
    pub irreversible_step_lock: u8,
    /// Current aperture contraction multiplier in Q0.7 (`0..=128`, initial `128 = 1.0`, `offset 33`).
    pub contraction_q7: u8,
    /// Most recent `ENVELOPE_BREACH_*` bitmask (`offset 34`).
    pub breach_mask: u8,
    /// Consecutive calm dwell counter reserved for online baseline tracking resume (`offset 35`).
    pub ema_resume_dwell: u8,
    /// Bitwise OR of `ENVELOPE_FLAG_*` status bits (`offset 36`).
    pub envelope_flags: u8,
    /// Explicit zero-initialized cacheline padding (`offset 37..64`).
    pub reserved: [u8; 27],
}

const _: () = {
    assert!(size_of::<EnvelopeState>() == 64);
    assert!(align_of::<EnvelopeState>() == 64);
    assert!(offset_of!(EnvelopeState, baseline_pinned_compute_mask) == 0);
    assert!(offset_of!(EnvelopeState, pinned_compute_mask) == 8);
    assert!(offset_of!(EnvelopeState, violation_epochs) == 16);
    assert!(offset_of!(EnvelopeState, calm_epochs) == 20);
    assert!(offset_of!(EnvelopeState, baseline_pressure_q15) == 24);
    assert!(offset_of!(EnvelopeState, last_pressure_q15) == 26);
    assert!(offset_of!(EnvelopeState, pinned_aperture_mask) == 28);
    assert!(offset_of!(EnvelopeState, pinned_slot_mask) == 29);
    assert!(offset_of!(EnvelopeState, pinned_action_mode_mask) == 30);
    assert!(offset_of!(EnvelopeState, unobserved_locked) == 31);
    assert!(offset_of!(EnvelopeState, irreversible_step_lock) == 32);
    assert!(offset_of!(EnvelopeState, contraction_q7) == 33);
    assert!(offset_of!(EnvelopeState, breach_mask) == 34);
    assert!(offset_of!(EnvelopeState, ema_resume_dwell) == 35);
    assert!(offset_of!(EnvelopeState, envelope_flags) == 36);
    assert!(offset_of!(EnvelopeState, reserved) == 37);
};

// SAFETY: `EnvelopeState` is `#[repr(C, align(64))]`, exact size 64 bytes with zero padding.
unsafe impl Pod for EnvelopeState {}

impl Default for EnvelopeState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl EnvelopeState {
    /// Constructs a nominal 64-byte [`EnvelopeState`] with unit contraction (`128`) and
    /// canonical immutable host pin floors.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            baseline_pinned_compute_mask: HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
            pinned_compute_mask: HOST_IMMUTABLE_PINNED_COMPUTE_MASK,
            violation_epochs: 0,
            calm_epochs: 0,
            baseline_pressure_q15: 0,
            last_pressure_q15: 0,
            pinned_aperture_mask: DEFAULT_PINNED_APERTURE_MASK_U8,
            pinned_slot_mask: DEFAULT_PINNED_SLOT_MASK_U8,
            pinned_action_mode_mask: ACTION_MODE_PHASE_REPULSE,
            unobserved_locked: 0,
            irreversible_step_lock: 0,
            contraction_q7: 128,
            breach_mask: ENVELOPE_BREACH_NONE,
            ema_resume_dwell: 0,
            envelope_flags: 0,
            reserved: [0; 27],
        }
    }

    /// Validates numeric range and flag bitmask invariants on [`EnvelopeState`] branchlessly.
    #[inline(always)]
    #[must_use]
    pub const fn validate_invariants(&self) -> bool {
        (self.contraction_q7 <= 128)
            & (self.baseline_pressure_q15 <= ENVELOPE_PRESSURE_Q15_ONE)
            & (self.last_pressure_q15 <= ENVELOPE_PRESSURE_Q15_ONE)
            & ((self.breach_mask & !ENVELOPE_BREACH_KNOWN_MASK) == 0)
            & ((self.envelope_flags & !ENVELOPE_FLAG_KNOWN_MASK) == 0)
    }

    /// Packs the active `ENVELOPE_LOCK_*` bitmask (`UNOBSERVED | IRREVERSIBLE | FAULT_ISOLATED`)
    /// from latched lock fields and `breach_mask`.
    #[inline(always)]
    #[must_use]
    pub const fn lock_flags(&self) -> u8 {
        let is_fault = (self.breach_mask & ENVELOPE_BREACH_UNDEFINED_FAULT) != 0;
        branchless_select_u8(self.unobserved_locked != 0, ENVELOPE_LOCK_UNOBSERVED, 0)
            | branchless_select_u8(
                self.irreversible_step_lock != 0,
                ENVELOPE_LOCK_IRREVERSIBLE,
                0,
            )
            | branchless_select_u8(is_fault, ENVELOPE_LOCK_FAULT_ISOLATED, 0)
    }

    /// Monotonically raises both permanent baseline and active compute pin masks, always
    /// enforcing [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
    #[inline(always)]
    pub const fn apply_host_pin_floor(&mut self, required_pinned_mask: u64) {
        let floor = required_pinned_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.baseline_pinned_compute_mask |= floor;
        self.pinned_compute_mask |= floor;
    }

    /// Replaces `baseline_pinned_compute_mask` (e.g., on geometry rebind) and synchronizes
    /// `pinned_compute_mask` while never dropping [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
    #[inline(always)]
    pub const fn set_baseline_pin_mask(&mut self, baseline_mask: u64) {
        let safe_base = baseline_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.baseline_pinned_compute_mask = safe_base;
        self.pinned_compute_mask = safe_base;
    }

    /// Arms a transient turn-local compute guard pin (`active_mask`) without polluting
    /// `baseline_pinned_compute_mask`.
    #[inline(always)]
    pub const fn arm_transient_guard_pin(&mut self, active_mask: u64) {
        self.pinned_compute_mask |=
            active_mask | self.baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
    }

    /// Restores `pinned_compute_mask` strictly to `baseline_pinned_compute_mask` plus
    /// [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`], clearing transient turn-local pins.
    #[inline(always)]
    pub const fn restore_baseline_pin_mask(&mut self) {
        self.pinned_compute_mask =
            self.baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
    }

    /// Pins additional compute slots in `pinned_compute_mask`.
    #[inline(always)]
    pub const fn pin_compute_slots(&mut self, slot_mask: u64) {
        self.pinned_compute_mask |= slot_mask;
    }

    /// Unpins transient compute slots from `pinned_compute_mask` while preserving the permanent
    /// baseline pin floor (`baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK`).
    #[inline(always)]
    pub const fn unpin_compute_slots(&mut self, slot_mask: u64) {
        let floor = self.baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.pinned_compute_mask = (self.pinned_compute_mask & !slot_mask) | floor;
    }

    /// Physical `pinned` isolation barrier ensuring sibling modulation never mutates pinned apertures.
    #[inline(always)]
    #[must_use]
    pub const fn mask_ego_aperture_edits(
        &self,
        prev_aperture_mask: u32,
        next_aperture_mask: u32,
    ) -> u32 {
        let pinned = self.pinned_aperture_mask as u32;
        let prev_cut = BitCut32::split(prev_aperture_mask, pinned);
        let next_cut = BitCut32::split(next_aperture_mask, pinned);
        next_cut.outside | prev_cut.inside
    }

    /// Primitive `pinned` isolation helper returning `(full_lock, effective_slot, effective_modes)`
    /// without depending on upper-layer `GatewayFrame`.
    #[inline(always)]
    #[must_use]
    pub const fn isolate_slot_and_modes_branchless(
        &self,
        base_slot: u8,
        base_modes: u8,
        ego_slot: u8,
        ego_modes: u8,
    ) -> (bool, u8, u8) {
        let masked_base = base_slot & ATTRACTOR_SLOT_ID_MASK;
        let slot_is_pinned = BitCut8::split(self.pinned_slot_mask, 1u8 << masked_base).inside != 0;
        let eff_slot = branchless_select_u8(slot_is_pinned, base_slot, ego_slot);
        let preserved_modes = BitCut8::split(base_modes, self.pinned_action_mode_mask).inside;
        let eff_modes = ego_modes | preserved_modes;
        let full_lock = (self.unobserved_locked != 0) | (self.irreversible_step_lock != 0);
        (full_lock, eff_slot, eff_modes)
    }

    /// Resets transient counters, locks, and contraction to nominal defaults while preserving the
    /// permanent host compute pin floor (`baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK`).
    #[inline(always)]
    pub const fn reset_preserving_pin_floor(&mut self) {
        let floor = self.baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.baseline_pinned_compute_mask = floor;
        self.pinned_compute_mask = floor;
        self.violation_epochs = 0;
        self.calm_epochs = 0;
        self.baseline_pressure_q15 = 0;
        self.last_pressure_q15 = 0;
        self.pinned_aperture_mask = DEFAULT_PINNED_APERTURE_MASK_U8;
        self.pinned_slot_mask = DEFAULT_PINNED_SLOT_MASK_U8;
        self.pinned_action_mode_mask = ACTION_MODE_PHASE_REPULSE;
        self.unobserved_locked = 0;
        self.irreversible_step_lock = 0;
        self.contraction_q7 = 128;
        self.breach_mask = ENVELOPE_BREACH_NONE;
        self.ema_resume_dwell = 0;
        self.envelope_flags = 0;
        self.reserved = [0; 27];
    }

    /// Evaluates one branchless safety-envelope step (`0 alloc`, `0 fdiv`) over `bounds` and
    /// `signal`, updating contraction, breach/calm counters, and monotonic pin floors.
    ///
    /// - Immutable host pin floors (`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`, `bounds.immutable_*`)
    ///   are strictly monotonic and never relaxed.
    /// - Terminal `Undefined` faults fail closed immediately (`contraction_q7 = 0`,
    ///   `ENVELOPE_LOCK_FAULT_ISOLATED`).
    /// - Bounded pressure, chatter, or exterior violations contract `contraction_q7` down to
    ///   `bounds.min_aperture_q7` and increment `violation_epochs`.
    /// - Violation-free observed steps recover `contraction_q7` monotonically up to `128`.
    #[inline]
    #[must_use]
    pub fn step(&mut self, bounds: &EnvelopeBounds, signal: &BoundarySignal) -> EnvelopeFeedback {
        let compute_floor = self.baseline_pinned_compute_mask
            | bounds.immutable_compute_pin_mask
            | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
        self.baseline_pinned_compute_mask = compute_floor;
        self.pinned_compute_mask |= compute_floor;
        self.pinned_aperture_mask |= bounds.immutable_aperture_pin_mask;
        self.pinned_slot_mask |= bounds.immutable_slot_pin_mask;
        self.pinned_action_mode_mask |= bounds.immutable_action_pin_mask;

        let is_undefined = signal.is_fault();
        let pressure_breach =
            (bounds.pressure_ceiling_q15 > 0) & (signal.pressure_q15 > bounds.pressure_ceiling_q15);
        let chatter_breach =
            (bounds.max_chatter_q7 > 0) & (signal.chatter_q7 > bounds.max_chatter_q7);
        let exterior_breach = signal.has_exterior_alert();

        let breach_mask = branchless_select_u8(pressure_breach, ENVELOPE_BREACH_PRESSURE, 0)
            | branchless_select_u8(chatter_breach, ENVELOPE_BREACH_CHATTER, 0)
            | branchless_select_u8(exterior_breach, ENVELOPE_BREACH_EXTERIOR, 0)
            | branchless_select_u8(is_undefined, ENVELOPE_BREACH_UNDEFINED_FAULT, 0);
        let any_breach = breach_mask != 0;
        let unobserved_zero = (!is_undefined)
            & (!any_breach)
            & signal.partition.is_unobserved_aperture_zero()
            & (!signal.is_observed());

        let cur = self.contraction_q7.min(128);
        let min_floor = bounds.min_aperture_q7.min(128);
        let c_gain = bounds.contraction_gain_q7.clamp(1, 128);
        let r_gain = bounds.recovery_gain_q7.clamp(1, 128);

        let dec_raw = cur.saturating_sub(c_gain);
        let contracted_bounded = branchless_select_u8(dec_raw < min_floor, min_floor, dec_raw);
        let sig_cap = signal.contraction_q7.min(128);
        let capped_bounded = branchless_select_u8(
            sig_cap < contracted_bounded,
            branchless_select_u8(sig_cap < min_floor, min_floor, sig_cap),
            contracted_bounded,
        );
        let recovered_raw = cur.saturating_add(r_gain);
        let recovered = branchless_select_u8(recovered_raw > 128, 128, recovered_raw);
        let calm_next = branchless_select_u8(unobserved_zero, cur, recovered);
        let non_fault_next = branchless_select_u8(any_breach, capped_bounded, calm_next);
        self.contraction_q7 = branchless_select_u8(is_undefined, 0, non_fault_next);

        self.violation_epochs = self.violation_epochs.saturating_add(u32::from(any_breach));
        let next_calm = self.calm_epochs.saturating_add(u32::from(!unobserved_zero));
        self.calm_epochs = branchless_select_u32(any_breach, 0, next_calm);
        self.last_pressure_q15 =
            branchless_select_u16(is_undefined, self.last_pressure_q15, signal.pressure_q15);
        self.breach_mask = breach_mask;

        let dwell_cap = bounds.ema_resume_dwell_target;
        let next_dwell_raw = self
            .ema_resume_dwell
            .saturating_add(u8::from(!unobserved_zero));
        let next_dwell =
            branchless_select_u8(next_dwell_raw > dwell_cap, dwell_cap, next_dwell_raw);
        self.ema_resume_dwell = branchless_select_u8(any_breach, 0, next_dwell);

        let is_contracted = self.contraction_q7 < 128;
        self.envelope_flags = branchless_select_u8(is_contracted, ENVELOPE_FLAG_CONTRACTED, 0)
            | branchless_select_u8(
                (!any_breach) & (!unobserved_zero) & is_contracted,
                ENVELOPE_FLAG_RECOVERY_ACTIVE,
                0,
            );

        let lock_flags = self.lock_flags()
            | branchless_select_u8(unobserved_zero, ENVELOPE_LOCK_UNOBSERVED, 0)
            | branchless_select_u8(
                signal.is_irreversible_intent(),
                ENVELOPE_LOCK_IRREVERSIBLE,
                0,
            );

        let is_stig = matches!(signal.attribution().primitive(), PrimitiveKind::Stigmergy);
        let attribution_tag = branchless_select_u8(
            is_stig,
            AttributionTag::STIGMERGY_ENVELOPE.raw(),
            AttributionTag::LIMINAL_ENVELOPE.raw(),
        );

        EnvelopeFeedback {
            effective_pinned_compute_mask: self.pinned_compute_mask,
            effective_pinned_aperture_mask: self.pinned_aperture_mask,
            effective_pinned_slot_mask: self.pinned_slot_mask,
            effective_pinned_action_mode_mask: self.pinned_action_mode_mask,
            contraction_q7: self.contraction_q7,
            breach_mask: self.breach_mask,
            lock_flags,
            attribution_tag,
            reserved: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liminal::{
        BOUNDARY_SIGNAL_FLAG_OBSERVED, DriverKind, evaluate_lacuna_partition_branchless,
    };

    #[test]
    fn test_envelope_layouts_and_pod_roundtrip() {
        assert_eq!(size_of::<EnvelopeBounds>(), 32);
        assert_eq!(align_of::<EnvelopeBounds>(), 32);
        assert_eq!(size_of::<EnvelopeFeedback>(), 16);
        assert_eq!(align_of::<EnvelopeFeedback>(), 16);
        assert_eq!(size_of::<EnvelopeState>(), 64);
        assert_eq!(align_of::<EnvelopeState>(), 64);

        let bounds = EnvelopeBounds::new();
        let bounds_rt = EnvelopeBounds::read_unaligned_from(bounds.as_bytes(), 0).unwrap();
        assert_eq!(bounds, bounds_rt);

        let mut state = EnvelopeState::new();
        state.apply_host_pin_floor(0x000A);
        let state_rt = EnvelopeState::read_unaligned_from(state.as_bytes(), 0).unwrap();
        assert_eq!(state, state_rt);

        let sig = BoundarySignal::new();
        let fb = state.step(&bounds, &sig);
        let fb_rt = EnvelopeFeedback::read_unaligned_from(fb.as_bytes(), 0).unwrap();
        assert_eq!(fb, fb_rt);
    }

    #[test]
    fn test_envelope_nominal_breach_recovery_and_pin_floor_invariants() {
        let bounds = EnvelopeBounds::new();
        let mut state = EnvelopeState::new();
        state.apply_host_pin_floor(0b0010_0100);
        assert_eq!(
            state.baseline_pinned_compute_mask,
            0b0010_0100 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );

        let tol = [10u8; 8];
        let calm_obs = [80, 72, 0, 0, 0, 0, 0, 0];
        let lp_calm = evaluate_lacuna_partition_branchless(0x03, &calm_obs, &tol, 40);
        let sig_calm = BoundarySignal::from_partition(lp_calm, 10_000, 512, 12, 1);

        // 1. Nominal observed step keeps contraction at 128 and increments calm_epochs.
        let fb0 = state.step(&bounds, &sig_calm);
        assert_eq!(fb0.contraction_q7, 128);
        assert_eq!(fb0.breach_mask, ENVELOPE_BREACH_NONE);
        assert_eq!(fb0.lock_flags, 0);
        assert_eq!(fb0.attribution_tag, AttributionTag::LIMINAL_ENVELOPE.raw());
        assert_eq!(
            AttributionTag(fb0.attribution_tag).driver(),
            DriverKind::Envelope
        );
        assert_eq!(state.calm_epochs, 1);
        assert_eq!(state.violation_epochs, 0);

        // 2. Pressure + chatter + exterior breach contracts toward min_aperture_q7 floor.
        let ext_obs = [80, 72, 0, 0, 92, 88, 0, 0];
        let lp_ext = evaluate_lacuna_partition_branchless(0x03, &ext_obs, &tol, 40);
        let mut sig_breach = BoundarySignal::from_partition(lp_ext, 31_000, 4_096, 96, 2);
        sig_breach.attribution_tag = AttributionTag::STIGMERGY_SYSTEM.raw();

        for _ in 0..8 {
            let fb = state.step(&bounds, &sig_breach);
            assert_eq!(
                fb.breach_mask,
                ENVELOPE_BREACH_PRESSURE | ENVELOPE_BREACH_CHATTER | ENVELOPE_BREACH_EXTERIOR
            );
            assert_eq!(fb.attribution_tag, AttributionTag::STIGMERGY_ENVELOPE.raw());
            assert!(fb.contraction_q7 >= bounds.min_aperture_q7);
        }
        assert_eq!(state.contraction_q7, bounds.min_aperture_q7);
        assert_eq!(state.violation_epochs, 8);
        assert_eq!(state.calm_epochs, 0);
        assert_ne!(state.envelope_flags & ENVELOPE_FLAG_CONTRACTED, 0);

        // 3. Calm recovery climbs smoothly back to 128 while never relaxing pinned masks.
        for _ in 0..20 {
            let fb = state.step(&bounds, &sig_calm);
            assert_eq!(
                fb.effective_pinned_compute_mask
                    & (0b0010_0100 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK),
                0b0010_0100 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
            );
        }
        assert_eq!(state.contraction_q7, 128);
        assert_eq!(state.calm_epochs, 20);
        assert_eq!(state.envelope_flags, 0);

        // 4. Undefined fault fails closed immediately to contraction_q7 == 0 and FAULT_ISOLATED.
        let fault_obs = [80, 0x80, 0, 0, 0, 0, 0, 0];
        let lp_fault = evaluate_lacuna_partition_branchless(0x03, &fault_obs, &tol, 40);
        let sig_fault = BoundarySignal::from_partition(lp_fault, 25_000, 0, 0, 3);
        let fb_fault = state.step(&bounds, &sig_fault);
        assert_eq!(fb_fault.contraction_q7, 0);
        assert_ne!(fb_fault.breach_mask & ENVELOPE_BREACH_UNDEFINED_FAULT, 0);
        assert_ne!(fb_fault.lock_flags & ENVELOPE_LOCK_FAULT_ISOLATED, 0);

        // 5. Irreversible intent + latched locks propagate cleanly and reset preserves pin floor.
        state.irreversible_step_lock = 1;
        state.unobserved_locked = 1;
        let mut sig_irrev = sig_calm.with_irreversible_intent(true);
        sig_irrev.signal_flags |= BOUNDARY_SIGNAL_FLAG_OBSERVED;
        let fb_locked = state.step(&bounds, &sig_irrev);
        assert_ne!(fb_locked.lock_flags & ENVELOPE_LOCK_IRREVERSIBLE, 0);
        assert_ne!(fb_locked.lock_flags & ENVELOPE_LOCK_UNOBSERVED, 0);

        state.reset_preserving_pin_floor();
        assert_eq!(state.contraction_q7, 128);
        assert_eq!(state.violation_epochs, 0);
        assert_eq!(state.unobserved_locked, 0);
        assert_eq!(state.irreversible_step_lock, 0);
        assert_eq!(
            state.baseline_pinned_compute_mask,
            0b0010_0100 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
        assert_eq!(
            state.pinned_compute_mask,
            0b0010_0100 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
    }
}
