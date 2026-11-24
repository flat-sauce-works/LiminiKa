// name: src/core/src/drivers/spine/state.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Mechanical execution spine state (`SystemSpineState`, 64 B), abstract decode signal boundary
//! ([`SpineDecodeSignal`], [`SpineDecodeInput`], [`SpineStepPlan`]), primitive `pinned` isolation
//! logic (`isolate_slot_and_modes_branchless`), and branchless I-Frame / P-Frame transition
//! evaluation (`evaluate_frame_transition_branchless`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]

use core::mem::{align_of, offset_of, size_of};

use crate::drivers::spine::compute_pool::{
    ComputeDispatchMask, SystemPressureReading, TokenBypassObservation,
};
use crate::foundation::{Pod, ACTION_MODE_PHASE_REPULSE, ATTRACTOR_SLOT_ID_MASK};
use crate::liminal::{
    branchless_select_u8, sensitivity_4p1mp_q7, AttributionTag, BitCut32, BitCut64, BitCut8,
    Lacuna, THRESHOLD_CONTRADICTION_GUARD, THRESHOLD_SYSTEM_CONSTRAINT,
};
use crate::stigmergy::{ATTRACTOR_SLOT_REPULSION, ATTRACTOR_SLOT_SYSTEM_BASELINE};

/// Frame representation kind in I-Frame / P-Frame steering and timeline encoding.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameKind {
    /// Complete keyframe anchor (I-Frame) establishing a reference gauge origin (`0`).
    #[default]
    Keyframe = 0,
    /// Incremental differential residual (P-Frame) relative to a prior keyframe (`1`).
    Delta = 1,
}

/// Action directive emitted by [`SystemSpineState::evaluate_frame_transition_branchless`].
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameDirective {
    /// Zero-aperture or stationary-ready skip; emit neither keyframe nor delta (`0`).
    #[default]
    Skip = 0,
    /// Mechanical pass-through locked by irreversible spine guards or OOD aperture locks (`1`).
    PassThrough = 1,
    /// Emit an incremental P-Frame differential update on allowed non-pinned granules (`2`).
    EmitDelta = 2,
    /// Re-anchor a fresh I-Frame keyframe on GOP expiry, exterior shift, new basin, or guard repulsion (`3`).
    ReAnchorKeyframe = 3,
    /// Reject non-finite [`Lacuna::Undefined`] measurement (`4`).
    RejectUndefined = 4,
}

/// Spine transition input flag (`bit 0`): current step is irreversible (e.g., prompt Prefill).
pub const SPINE_TRANSITION_FLAG_IRREVERSIBLE: u8 = 1 << 0;
/// Spine transition input flag (`bit 1`): contradiction guard or repulsion escape fired.
pub const SPINE_TRANSITION_FLAG_GUARD_REPULSION: u8 = 1 << 1;
/// Spine transition input flag (`bit 2`): a new attractor basin crystallized on this step.
pub const SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED: u8 = 1 << 2;
/// Spine transition input flag (`bit 3`): stationary self-reference shortcut is ready.
pub const SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY: u8 = 1 << 3;

/// Compact 8-byte (`#[repr(C, align(8))]`) transition input evaluated branchlessly by
/// [`SystemSpineState::evaluate_frame_transition_branchless`].
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpineFrameTransitionInput {
    /// Encoded [`Lacuna`] discriminant (`0=Unobserved`, `1=SelfRef`, `2=Exterior`, `3=Undefined`, `offset 0`).
    pub lacuna_code: u8,
    /// Quantized L1 phase shift relative to the reference keyframe in Q0.7 (`0..=127`, `offset 1`).
    pub phase_delta_q7: u8,
    /// Re-anchor phase delta threshold in Q0.7 (`0` disables delta-based re-anchoring, `offset 2`).
    pub reanchor_delta_thresh_q7: u8,
    /// Number of delta steps elapsed since the last I-Frame keyframe (`offset 3`).
    pub steps_since_keyframe: u8,
    /// Maximum GOP length before re-anchoring (`0` disables GOP-length expiry, `offset 4`).
    pub max_gop_steps: u8,
    /// Candidate 8-granule modification mask prior to `pinned_slot_mask` isolation (`offset 5`).
    pub candidate_granule_mask: u8,
    /// Bitwise OR of `SPINE_TRANSITION_FLAG_*` conditions (`offset 6`).
    pub flags: u8,
    /// Explicit C alignment padding (`offset 7`).
    pub reserved: u8,
}

const _: () = {
    assert!(size_of::<SpineFrameTransitionInput>() == 8);
    assert!(align_of::<SpineFrameTransitionInput>() == 8);
};

impl SpineFrameTransitionInput {
    /// Builds a [`SpineFrameTransitionInput`] branchlessly from a 16-byte [`crate::liminal::LacunaPartition8`].
    #[inline(always)]
    #[must_use]
    pub const fn from_lacuna_partition(
        partition: &crate::liminal::LacunaPartition8,
        phase_delta_q7: u8,
        reanchor_delta_thresh_q7: u8,
        steps_since_keyframe: u8,
        max_gop_steps: u8,
        flags: u8,
    ) -> Self {
        let mut lacuna_code = Lacuna::Unobserved.code();
        if partition.self_ref_mask != 0 {
            lacuna_code = Lacuna::SelfRef.code();
        }
        if partition.exterior_mask != 0 {
            lacuna_code = Lacuna::Exterior.code();
        }
        if partition.undefined_mask != 0 {
            lacuna_code = Lacuna::Undefined.code();
        }
        Self {
            lacuna_code,
            phase_delta_q7,
            reanchor_delta_thresh_q7,
            steps_since_keyframe,
            max_gop_steps,
            candidate_granule_mask: partition.self_ref_mask | partition.exterior_mask,
            flags,
            reserved: 0,
        }
    }
}

// SAFETY: `SpineFrameTransitionInput` is `#[repr(C, align(8))]` of exact size 8 bytes with primitive `u8` fields only.
unsafe impl Pod for SpineFrameTransitionInput {}

// 3. Spine Decode Signal Boundary (`SpineDecodeSignal` / `SpineDecodeInput` / `SpineStepPlan`)
// ===================================================================

/// Bitmask of valid `SPINE_BYPASS_VETO_*` flags (`0x07`).
pub const SPINE_BYPASS_VETO_KNOWN_MASK: u8 = SPINE_BYPASS_VETO_GUARD_ACTIVE
    | SPINE_BYPASS_VETO_REPULSION_ACTIVE
    | SPINE_BYPASS_VETO_PINNED_LOCK;

/// No spine bypass veto active (`0x00`).
pub const SPINE_BYPASS_VETO_NONE: u8 = 0x00;
/// Bypass vetoed because a contradiction guard boundary is active (`1 << 0`).
pub const SPINE_BYPASS_VETO_GUARD_ACTIVE: u8 = 1 << 0;
/// Bypass vetoed because phase-conjugate repulsion is active (`1 << 1`).
pub const SPINE_BYPASS_VETO_REPULSION_ACTIVE: u8 = 1 << 1;
/// Bypass vetoed because an irreversible step lock or pinned compute mask is active (`1 << 2`).
pub const SPINE_BYPASS_VETO_PINNED_LOCK: u8 = 1 << 2;

/// Minimum host-immutable compute slot bitmask (`bit 0`) preserved across untrusted snapshot restores.
pub const HOST_IMMUTABLE_PINNED_COMPUTE_MASK: u64 = 1u64 << 0;

/// Spine control flag (`bit 0`): active turn holds a contradiction-guard compute pin.
pub const SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE: u8 = 1 << 0;
/// Spine control flag (`bit 1`): transient contradiction-guard compute pin was safely relaxed.
pub const SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED: u8 = 1 << 1;
/// Spine control flag (`bit 2`): `PREV_WAS_ABSTRACT` residency tracking bit is initialized for this turn.
pub const SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID: u8 = 1 << 2;
/// Spine control flag (`bit 3`): previous decode step ran in abstract (`whole_token_bypass_active`) residency.
pub const SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT: u8 = 1 << 3;
/// Spine control flag (`bit 4`): pre-inference whole-token bypass evaluation is enabled.
pub const SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED: u8 = 1 << 4;
/// Bitmask of all defined `SPINE_CONTROL_FLAG_*` bits (`0x1F`).
pub const SPINE_CONTROL_KNOWN_FLAGS: u8 = SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE
    | SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED
    | SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID
    | SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT
    | SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED;

/// No special spine decode step events emitted (`0x00`).
pub const SPINE_STEP_EVENT_NONE: u8 = 0x00;
/// Step event flag (`bit 0`): residency transitioned between abstract and concrete mode.
pub const SPINE_STEP_EVENT_BOUNDARY_CHURN: u8 = 1 << 0;
/// Step event flag (`bit 1`): logits observation was skipped (`Lacuna::Unobserved`) on this step.
pub const SPINE_STEP_EVENT_LOGITS_OBSERVE_SKIPPED: u8 = 1 << 1;
/// Step event flag (`bit 2`): transient contradiction-guard pin relaxed cleanly.
pub const SPINE_STEP_EVENT_TRANSIENT_GUARD_RELAXED: u8 = 1 << 2;
/// Step event flag (`bit 3`): relaxed contradiction-guard pin re-escalated immediately on instability.
pub const SPINE_STEP_EVENT_GUARD_REESCALATED: u8 = 1 << 3;
/// Step event flag (`bit 4`): exit entropy surge or [`Lacuna::Undefined`] triggered instability recovery.
pub const SPINE_STEP_EVENT_INSTABILITY_RECOVERY: u8 = 1 << 4;

/// [`SpineDecodeInput`] flag (`bit 0`): phase-conjugate repulsion was active on the completed step.
pub const SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE: u8 = 1 << 0;
/// [`SpineDecodeInput`] flag (`bit 1`): greedy sampling (`temperature <= 1e-6`) is active.
pub const SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING: u8 = 1 << 1;
/// [`SpineDecodeInput`] flag (`bit 2`): pre-inference whole-token bypass evaluation is enabled.
pub const SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED: u8 = 1 << 2;
/// [`SpineDecodeInput`] flag (`bit 3`): `exit_reward_override_q7` overrides automatic reward selection.
pub const SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE: u8 = 1 << 3;
/// Canonical alias for [`SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE`] (`bit 3`).
pub const SPINE_DECODE_FLAG_REWARD_OVERRIDE: u8 = SPINE_DECODE_INPUT_FLAG_REWARD_OVERRIDE;

/// Sentinel `u8` (`0xFF`) delegating sensitivity or reward override selection to core spine defaults.
pub const SPINE_SENTINEL_AUTO_U8: u8 = 0xFF;

/// Unit scale (`1.0`) for normalized exit entropy in Q0.15 (`32_768`).
pub const SPINE_ENTROPY_Q15_ONE: u16 = 32_768;
/// Stationary exit-entropy upper bound (`0.78` in Q0.15) separating `SelfRef` from `Exterior`.
pub const SPINE_ENTROPY_CALM_THRESH_Q15: u16 = 25_559;
/// Maximum observed exit entropy (`0.72` in Q0.15) allowed for transient contradiction-guard relaxation.
pub const SPINE_ENTROPY_GUARD_RELAX_MAX_Q15: u16 = 23_593;
/// Exit entropy surge cutoff (`0.85` in Q0.15) triggering immediate full-layer instability recovery.
pub const SPINE_ENTROPY_SURGE_THRESH_Q15: u16 = 27_853;
/// Low-entropy reward threshold (`0.55` in Q0.15) splitting `56` vs `96` Q0.7 exit pheromone reward.
pub const SPINE_ENTROPY_REWARD_LOW_THRESH_Q15: u16 = 18_022;
/// Baseline entropy floor (`0.15` in Q0.15) subtracted when feeding `DistortionTension`.
pub const SPINE_ENTROPY_TENSION_FLOOR_Q15: u16 = 4_915;
/// Minimum dynamic logits observation stride (`2`) when dynamic subsampling (`max_observe_stride > 1`) is active.
pub const SPINE_MIN_DYNAMIC_OBSERVE_STRIDE: u8 = 2;

const SPINE_ENTROPY_CALM_THRESH_F32: f32 = 0.78;

/// Logits D2H observation directive for the upcoming decode forward pass.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpineLogitsObserveMode {
    /// Transfer full logits to host CPU for entropy observation (`0`).
    #[default]
    ObserveFullLogits = 0,
    /// Keep logits on device and fetch only the greedy argmax token ID (`1`).
    SkipToDeviceArgmax = 1,
}

impl SpineLogitsObserveMode {
    /// Returns the compact `u8` discriminant (`0` = `ObserveFullLogits`, `1` = `SkipToDeviceArgmax`).
    #[inline(always)]
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }

    /// Decodes a `u8` discriminant, defaulting fail-safe to [`Self::ObserveFullLogits`].
    #[inline(always)]
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::SkipToDeviceArgmax,
            _ => Self::ObserveFullLogits,
        }
    }
}

/// Abstract zero-allocation observation boundary consumed by `RuntimeContext::step_decode_control`
/// via static generics (monomorphization, `0 alloc`, `0` host-runtime coupling).
pub trait SpineDecodeSignal {
    /// Returns the external `+ System` Gauge-Zero pressure reading for the next step.
    fn pressure_reading(&self) -> SystemPressureReading;

    /// Returns the 4-state [`Lacuna`] classification of exit-logit entropy for the completed step.
    fn exit_entropy_lacuna(&self) -> Lacuna;

    /// Returns the observed exit-logit entropy in Q0.15 (`0..=32768`) when `exit_entropy_lacuna().observed()`.
    fn observed_exit_entropy_q15(&self) -> u16;

    /// Returns the $4p(1-p)$ boundary sensitivity in Q0.7 (`0..=127`), or [`SPINE_SENTINEL_AUTO_U8`]
    /// (`0xFF`) to fall back to the active context's `last_lacuna_partition.boundary_sensitivity_q7`.
    fn boundary_sensitivity_q7(&self) -> u8;

    /// Returns the 0-based index of the decode step that just completed (`step_idx`).
    fn decode_step_index(&self) -> u16;

    /// Returns `true` when phase-conjugate repulsion fired on the completed step.
    fn repulsion_active(&self) -> bool;

    /// Returns `true` when decoding runs deterministically (`temperature <= 1e-6`).
    fn greedy_sampling(&self) -> bool;

    /// Minimum consecutive calm decode steps required before re-engaging whole-token bypass (`default = 2`).
    #[inline(always)]
    fn min_calm_dwell(&self) -> u8 {
        2
    }

    /// Optional Q0.7 pheromone exit reward override (`0..=127`), or [`SPINE_SENTINEL_AUTO_U8`] (`0xFF`)
    /// to compute the reward inside core from the latest observed entropy (`< 0.55` -> `56`, else `96`).
    #[inline(always)]
    fn exit_reward_override_q7(&self) -> u8 {
        SPINE_SENTINEL_AUTO_U8
    }

    /// Returns `true` when pre-inference whole-token steering bypass is permitted for this step.
    #[inline(always)]
    fn pre_bypass_enabled(&self) -> bool {
        true
    }
}

/// Compact 32-byte (`#[repr(C, align(16))]`, zero implicit padding) concrete [`Pod`] implementing
/// [`SpineDecodeSignal`] with 1:1 trait parity for host FFI and adapter callers (`lmnk_spine_decode_input_t`).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpineDecodeInput {
    /// External `+ System` Gauge-Zero pressure reading (`offset 0..8`).
    pub pressure: SystemPressureReading,
    /// Quantized exit-logit entropy in Q0.15 (`0..=32768`, `offset 8..10`).
    pub observed_exit_entropy_q15: u16,
    /// 0-based decode step index that just completed (`offset 10..12`).
    pub decode_step_index: u16,
    /// Encoded [`Lacuna`] classification (`0=Unobserved`, `1=SelfRef`, `2=Exterior`, `3=Undefined`, `offset 12`).
    pub exit_entropy_lacuna_code: u8,
    /// Boundary sensitivity in Q0.7 (`0..=127`), or [`SPINE_SENTINEL_AUTO_U8`] (`0xFF`, `offset 13`).
    pub boundary_sensitivity_q7: u8,
    /// Minimum calm dwell tokens before whole-token bypass (`0` uses default `2`, `offset 14`).
    pub min_calm_dwell: u8,
    /// Bitwise OR of `SPINE_DECODE_INPUT_FLAG_*` flags (`offset 15`).
    pub flags: u8,
    /// Explicit Q0.7 exit pheromone reward (`0..=127`) honored when [`SPINE_DECODE_FLAG_REWARD_OVERRIDE`] is set (`offset 16`).
    pub exit_reward_override_q7: u8,
    /// Explicit zero-initialized reserved space for future ABI extensions (`offset 17..32`).
    pub reserved: [u8; 15],
}

const _: () = {
    assert!(size_of::<SpineDecodeInput>() == 32);
    assert!(align_of::<SpineDecodeInput>() == 16);
    assert!(offset_of!(SpineDecodeInput, pressure) == 0);
    assert!(offset_of!(SpineDecodeInput, observed_exit_entropy_q15) == 8);
    assert!(offset_of!(SpineDecodeInput, decode_step_index) == 10);
    assert!(offset_of!(SpineDecodeInput, exit_entropy_lacuna_code) == 12);
    assert!(offset_of!(SpineDecodeInput, boundary_sensitivity_q7) == 13);
    assert!(offset_of!(SpineDecodeInput, min_calm_dwell) == 14);
    assert!(offset_of!(SpineDecodeInput, flags) == 15);
    assert!(offset_of!(SpineDecodeInput, exit_reward_override_q7) == 16);
    assert!(offset_of!(SpineDecodeInput, reserved) == 17);
};

// SAFETY: `SpineDecodeInput` is `#[repr(C, align(16))]`, exact size 32 bytes with zero implicit padding.
unsafe impl Pod for SpineDecodeInput {}

impl SpineDecodeInput {
    /// Constructs a [`SpineDecodeInput`] from an observed floating-point exit-logit entropy,
    /// encapsulating the core [`Lacuna`] classification (`NaN`/`±Inf` -> `Undefined`,
    /// `< 0.78` -> `SelfRef`, `>= 0.78` -> `Exterior`) inside `liminika-core`.
    #[inline]
    #[must_use]
    pub fn from_observed_entropy_f32(
        pressure: SystemPressureReading,
        decode_step_index: u16,
        exit_entropy: f32,
        boundary_sensitivity_q7: u8,
        repulsion_active: bool,
        greedy_sampling: bool,
    ) -> Self {
        let (lacuna, ent_q15) = if !exit_entropy.is_finite() {
            (Lacuna::Undefined, 0)
        } else {
            let clamped = exit_entropy.clamp(0.0, 1.0);
            let q15 = ((clamped * f32::from(SPINE_ENTROPY_Q15_ONE) + 0.5) as u16)
                .min(SPINE_ENTROPY_Q15_ONE);
            let lac = if clamped < SPINE_ENTROPY_CALM_THRESH_F32 {
                Lacuna::SelfRef
            } else {
                Lacuna::Exterior
            };
            (lac, q15)
        };
        let mut flags = SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED;
        if repulsion_active {
            flags |= SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE;
        }
        if greedy_sampling {
            flags |= SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING;
        }
        Self {
            pressure,
            observed_exit_entropy_q15: ent_q15,
            decode_step_index,
            exit_entropy_lacuna_code: lacuna.code(),
            boundary_sensitivity_q7,
            min_calm_dwell: 2,
            flags,
            exit_reward_override_q7: 0,
            reserved: [0; 15],
        }
    }

    /// Constructs a [`SpineDecodeInput`] representing a skipped logits observation
    /// ([`Lacuna::Unobserved`], zero-aperture non-observation).
    #[inline(always)]
    #[must_use]
    pub const fn unobserved(
        pressure: SystemPressureReading,
        decode_step_index: u16,
        boundary_sensitivity_q7: u8,
        repulsion_active: bool,
        greedy_sampling: bool,
    ) -> Self {
        let mut flags = SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED;
        if repulsion_active {
            flags |= SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE;
        }
        if greedy_sampling {
            flags |= SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING;
        }
        Self {
            pressure,
            observed_exit_entropy_q15: 0,
            decode_step_index,
            exit_entropy_lacuna_code: Lacuna::Unobserved.code(),
            boundary_sensitivity_q7,
            min_calm_dwell: 2,
            flags,
            exit_reward_override_q7: 0,
            reserved: [0; 15],
        }
    }

    /// Sets whether pre-inference whole-token bypass evaluation is enabled for this step.
    #[inline(always)]
    #[must_use]
    pub const fn with_pre_bypass(mut self, enabled: bool) -> Self {
        if enabled {
            self.flags |= SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED;
        } else {
            self.flags &= !SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED;
        }
        self
    }

    /// Overrides the minimum consecutive calm dwell tokens required before whole-token bypass.
    #[inline(always)]
    #[must_use]
    pub const fn with_min_calm_dwell(mut self, min_calm_dwell: u8) -> Self {
        self.min_calm_dwell = if min_calm_dwell < 2 {
            2
        } else {
            min_calm_dwell
        };
        self
    }

    /// Arms [`SPINE_DECODE_FLAG_REWARD_OVERRIDE`] and sets the explicit Q0.7 exit pheromone reward
    /// clamped safely into `0..=127`.
    #[inline(always)]
    #[must_use]
    pub const fn with_exit_reward_override_q7(mut self, reward_q7: u8) -> Self {
        self.exit_reward_override_q7 = if reward_q7 > 127 { 127 } else { reward_q7 };
        self.flags |= SPINE_DECODE_FLAG_REWARD_OVERRIDE;
        self
    }

    /// Clears [`SPINE_DECODE_FLAG_REWARD_OVERRIDE`] so core derives the exit reward automatically.
    #[inline(always)]
    #[must_use]
    pub const fn without_exit_reward_override(mut self) -> Self {
        self.exit_reward_override_q7 = 0;
        self.flags &= !SPINE_DECODE_FLAG_REWARD_OVERRIDE;
        self
    }
}

impl SpineDecodeSignal for SpineDecodeInput {
    #[inline(always)]
    fn pressure_reading(&self) -> SystemPressureReading {
        self.pressure
    }

    #[inline(always)]
    fn exit_entropy_lacuna(&self) -> Lacuna {
        Lacuna::from_code(self.exit_entropy_lacuna_code)
    }

    #[inline(always)]
    fn observed_exit_entropy_q15(&self) -> u16 {
        self.observed_exit_entropy_q15.min(SPINE_ENTROPY_Q15_ONE)
    }

    #[inline(always)]
    fn boundary_sensitivity_q7(&self) -> u8 {
        self.boundary_sensitivity_q7
    }

    #[inline(always)]
    fn decode_step_index(&self) -> u16 {
        self.decode_step_index
    }

    #[inline(always)]
    fn repulsion_active(&self) -> bool {
        (self.flags & SPINE_DECODE_INPUT_FLAG_REPULSION_ACTIVE) != 0
    }

    #[inline(always)]
    fn greedy_sampling(&self) -> bool {
        (self.flags & SPINE_DECODE_INPUT_FLAG_GREEDY_SAMPLING) != 0
    }

    #[inline(always)]
    fn min_calm_dwell(&self) -> u8 {
        if self.min_calm_dwell < 2 {
            2
        } else {
            self.min_calm_dwell
        }
    }

    #[inline(always)]
    fn exit_reward_override_q7(&self) -> u8 {
        if (self.flags & SPINE_DECODE_FLAG_REWARD_OVERRIDE) != 0 {
            self.exit_reward_override_q7.min(127)
        } else {
            SPINE_SENTINEL_AUTO_U8
        }
    }

    #[inline(always)]
    fn pre_bypass_enabled(&self) -> bool {
        (self.flags & SPINE_DECODE_INPUT_FLAG_PRE_BYPASS_ENABLED) != 0
    }
}

/// Cacheline-aligned 64-byte (`#[repr(C, align(64))]`, zero implicit padding) pre-execution decode
/// step plan emitted by `RuntimeContext::step_decode_control` (`lmnk_spine_step_plan_t`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SpineStepPlan {
    /// Pre-execution compute dispatch plan for the next decode token (`offset 0..40`).
    pub dispatch: ComputeDispatchMask,
    /// Encoded [`SpineLogitsObserveMode`] governing logits fetch on the next decode token (`offset 40`).
    pub next_observe_mode: u8,
    /// Active `SPINE_BYPASS_VETO_*` bitmask evaluated for the next decode token (`offset 41`).
    pub veto_mask: u8,
    /// Bitwise OR of `SPINE_STEP_EVENT_*` transitions observed on this step (`offset 42`).
    pub event_flags: u8,
    /// Effective dynamic logits observation stride computed for the next decode token (`offset 43`).
    pub effective_observe_stride: u8,
    /// Explicit 64-byte cacheline alignment and future extension padding (`offset 44..64`).
    pub reserved_align: [u8; 20],
}

const _: () = {
    assert!(size_of::<SpineStepPlan>() == 64);
    assert!(align_of::<SpineStepPlan>() == 64);
    assert!(offset_of!(SpineStepPlan, dispatch) == 0);
    assert!(offset_of!(SpineStepPlan, next_observe_mode) == 40);
    assert!(offset_of!(SpineStepPlan, veto_mask) == 41);
    assert!(offset_of!(SpineStepPlan, event_flags) == 42);
    assert!(offset_of!(SpineStepPlan, effective_observe_stride) == 43);
    assert!(offset_of!(SpineStepPlan, reserved_align) == 44);
};

// SAFETY: `SpineStepPlan` is `#[repr(C, align(64))]`, exact size 64 bytes with zero implicit padding.
unsafe impl Pod for SpineStepPlan {}

impl SpineStepPlan {
    /// Returns [`SpineLogitsObserveMode`] decoded fail-safely from `self.next_observe_mode`.
    #[inline(always)]
    #[must_use]
    pub const fn next_observe_mode_enum(&self) -> SpineLogitsObserveMode {
        SpineLogitsObserveMode::from_code(self.next_observe_mode)
    }

    /// Returns `true` when the next decode token runs in whole-token steering bypass (`0` D2H syncs).
    #[inline(always)]
    #[must_use]
    pub const fn is_whole_token_bypass(&self) -> bool {
        self.dispatch.whole_token_steering_bypass != 0
    }
}

// ===================================================================
// 4. System Spine State & Physical `pinned` Isolation (64 B = 1 CL)
// ===================================================================

/// Mechanical execution spine (`64 B`, `#[repr(C, align(64))]`, `DriverKind::System` = I-Frame).
///
/// Owns physical `pinned` masks isolating irreversible steps, memory safety, OOD aperture-0 locks,
/// host permanent baseline pins (`baseline_pinned_compute_mask`), and decode bypass/stride SSOT
/// state so `EgoDifferentialNet` masks/XORs and transient guard pins cannot corrupt host invariants.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SystemSpineState {
    /// Physical `Liminal` aperture bits protected from Ego override (`offset 0..4`).
    pub pinned_aperture_mask: u32,
    /// Physical workbench slot bits protected from Ego deflection (`offset 4`).
    pub pinned_slot_mask: u8,
    /// Physical `ACTION_MODE_*` bits protected from Ego suppression (`offset 5`).
    pub pinned_action_mode_mask: u8,
    /// Non-zero when Unobserved boundary ($\overline{S \cup O}$) forces aperture-0 lock (`offset 6`).
    pub unobserved_locked: u8,
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
    /// Explicit C alignment padding for the `0..32` ABI-preserved prefix (`offset 28..32`).
    pub reserved_half0: [u8; 4],
    /// Permanent host compute slot pin floor protected from transient guard pin clearing (`offset 32..40`).
    pub baseline_pinned_compute_mask: u64,
    /// Latest finite observed exit-logit entropy in Q0.15 (`0..=32768`, `offset 40..42`).
    pub last_observed_entropy_q15: u16,
    /// Opening decode token hold window for transient guard pins (`0` = hold entire turn, `offset 42`).
    pub guard_hold_tokens: u8,
    /// Remaining concrete probe steps enforced after Slow-Loop `PassiveSaturation` (`offset 43`).
    pub passive_saturation_probe_steps: u8,
    /// Maximum logits observation stride (`1` = observe every token; `0` treated defensively as `1`, `offset 44`).
    pub max_observe_stride: u8,
    /// Consecutive unobserved whole-token bypass skips since last full-logits observation (`offset 45`).
    pub unobserved_bypass_run: u8,
    /// Consecutive concrete (`whole_token_bypass_active == 0`) decode steps in active turn (`offset 46`).
    pub concrete_run_steps: u8,
    /// Consecutive calm concrete decode steps eligible for transient guard relaxation (`offset 47`).
    pub calm_concrete_streak: u8,
    /// Bitwise OR of `SPINE_CONTROL_FLAG_*` state bits (`offset 48`).
    pub control_flags: u8,
    /// Most recent `SPINE_BYPASS_VETO_*` bitmask (`offset 49`).
    pub last_veto_mask: u8,
    /// Explicit C alignment padding (`offset 50..56`).
    pub reserved_mid: [u8; 6],
    /// Compute slot execution mask applied on the most recently dispatched step (`offset 56..64`).
    pub last_executed_mask: u64,
}

const _: () = {
    assert!(size_of::<SystemSpineState>() == 64);
    assert!(align_of::<SystemSpineState>() == 64);
    assert!(offset_of!(SystemSpineState, pinned_aperture_mask) == 0);
    assert!(offset_of!(SystemSpineState, pinned_slot_mask) == 4);
    assert!(offset_of!(SystemSpineState, pinned_action_mode_mask) == 5);
    assert!(offset_of!(SystemSpineState, unobserved_locked) == 6);
    assert!(offset_of!(SystemSpineState, irreversible_step_lock) == 7);
    assert!(offset_of!(SystemSpineState, pinned_compute_mask) == 8);
    assert!(offset_of!(SystemSpineState, gauge_anchor_epoch) == 16);
    assert!(offset_of!(SystemSpineState, slow_loop_epoch) == 20);
    assert!(offset_of!(SystemSpineState, whole_token_bypass_active) == 24);
    assert!(offset_of!(SystemSpineState, bypass_dwell_steps) == 25);
    assert!(offset_of!(SystemSpineState, bypass_cooldown_steps) == 26);
    assert!(offset_of!(SystemSpineState, attribution_tag) == 27);
    assert!(offset_of!(SystemSpineState, reserved_half0) == 28);
    assert!(offset_of!(SystemSpineState, baseline_pinned_compute_mask) == 32);
    assert!(offset_of!(SystemSpineState, last_observed_entropy_q15) == 40);
    assert!(offset_of!(SystemSpineState, guard_hold_tokens) == 42);
    assert!(offset_of!(SystemSpineState, passive_saturation_probe_steps) == 43);
    assert!(offset_of!(SystemSpineState, max_observe_stride) == 44);
    assert!(offset_of!(SystemSpineState, unobserved_bypass_run) == 45);
    assert!(offset_of!(SystemSpineState, concrete_run_steps) == 46);
    assert!(offset_of!(SystemSpineState, calm_concrete_streak) == 47);
    assert!(offset_of!(SystemSpineState, control_flags) == 48);
    assert!(offset_of!(SystemSpineState, last_veto_mask) == 49);
    assert!(offset_of!(SystemSpineState, reserved_mid) == 50);
    assert!(offset_of!(SystemSpineState, last_executed_mask) == 56);
};

impl Default for SystemSpineState {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpineState {
    /// Constructs a canonical [`SystemSpineState`] locking SystemConstraint, ContradictionGuard,
    /// and `ACTION_MODE_PHASE_REPULSE` in physical `pinned` storage with `max_observe_stride = 1`.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pinned_aperture_mask: THRESHOLD_SYSTEM_CONSTRAINT | THRESHOLD_CONTRADICTION_GUARD,
            pinned_slot_mask: (1u8 << ATTRACTOR_SLOT_SYSTEM_BASELINE)
                | (1u8 << ATTRACTOR_SLOT_REPULSION),
            pinned_action_mode_mask: ACTION_MODE_PHASE_REPULSE,
            unobserved_locked: 0,
            irreversible_step_lock: 0,
            pinned_compute_mask: 0,
            gauge_anchor_epoch: 0,
            slow_loop_epoch: 0,
            whole_token_bypass_active: 0,
            bypass_dwell_steps: 0,
            bypass_cooldown_steps: 0,
            attribution_tag: AttributionTag::LIMINAL_SYSTEM.raw(),
            reserved_half0: [0; 4],
            baseline_pinned_compute_mask: 0,
            last_observed_entropy_q15: 0,
            guard_hold_tokens: 0,
            passive_saturation_probe_steps: 0,
            max_observe_stride: 1,
            unobserved_bypass_run: 0,
            concrete_run_steps: 0,
            calm_concrete_streak: 0,
            control_flags: SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED,
            last_veto_mask: SPINE_BYPASS_VETO_NONE,
            reserved_mid: [0; 6],
            last_executed_mask: 0,
        }
    }

    /// Constructs a canonical [`SystemSpineState`] initialized with `guard_hold_tokens` (`0` = hold
    /// full turn) and `max_observe_stride` (`0` normalized defensively to `1`).
    #[inline(always)]
    #[must_use]
    pub const fn with_policy(guard_hold_tokens: u8, max_observe_stride: u8) -> Self {
        let mut spine = Self::new();
        spine.guard_hold_tokens = guard_hold_tokens;
        spine.set_max_observe_stride(max_observe_stride);
        spine
    }

    /// Raises both permanent host baseline pins (`baseline_pinned_compute_mask`) and active
    /// `pinned_compute_mask` so untrusted snapshots or transient guard resets cannot strip them.
    #[inline(always)]
    pub const fn apply_host_pin_floor(&mut self, required_pinned_mask: u64) {
        self.baseline_pinned_compute_mask |= required_pinned_mask;
        self.pinned_compute_mask |= required_pinned_mask;
    }

    /// Replaces `baseline_pinned_compute_mask` (used when model geometry rebinds) and updates
    /// `pinned_compute_mask` without dropping [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`].
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
        self.pinned_compute_mask |= active_mask | self.baseline_pinned_compute_mask;
    }

    /// Restores `pinned_compute_mask` strictly to `baseline_pinned_compute_mask` plus
    /// [`HOST_IMMUTABLE_PINNED_COMPUTE_MASK`], clearing any transient guard pin bits.
    #[inline(always)]
    pub const fn restore_baseline_pin_mask(&mut self) {
        self.pinned_compute_mask =
            self.baseline_pinned_compute_mask | HOST_IMMUTABLE_PINNED_COMPUTE_MASK;
    }

    /// Returns `true` when the active turn holds a contradiction-guard compute pin.
    #[inline(always)]
    #[must_use]
    pub const fn turn_guard_active(&self) -> bool {
        (self.control_flags & SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE) != 0
    }

    /// Sets or clears `SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE`.
    #[inline(always)]
    pub const fn set_turn_guard_active(&mut self, active: bool) {
        if active {
            self.control_flags |= SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE;
        } else {
            self.control_flags &= !SPINE_CONTROL_FLAG_TURN_GUARD_ACTIVE;
        }
    }

    /// Returns `true` when the active turn's transient contradiction-guard pin was relaxed.
    #[inline(always)]
    #[must_use]
    pub const fn is_transient_guard_relaxed(&self) -> bool {
        (self.control_flags & SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED) != 0
    }

    /// Sets or clears `SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED`.
    #[inline(always)]
    pub const fn set_transient_guard_relaxed(&mut self, relaxed: bool) {
        if relaxed {
            self.control_flags |= SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED;
        } else {
            self.control_flags &= !SPINE_CONTROL_FLAG_TRANSIENT_GUARD_RELAXED;
        }
    }

    /// Returns `true` when pre-inference whole-token bypass evaluation is enabled.
    #[inline(always)]
    #[must_use]
    pub const fn is_pre_bypass_enabled(&self) -> bool {
        (self.control_flags & SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED) != 0
    }

    /// Sets or clears `SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED`.
    #[inline(always)]
    pub const fn set_pre_bypass_enabled(&mut self, enabled: bool) {
        if enabled {
            self.control_flags |= SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED;
        } else {
            self.control_flags &= !SPINE_CONTROL_FLAG_PRE_BYPASS_ENABLED;
        }
    }

    /// Returns `max_observe_stride` clamped defensively to `>= 1` (safe even after zeroization).
    #[inline(always)]
    #[must_use]
    pub const fn effective_max_observe_stride(&self) -> u8 {
        if self.max_observe_stride == 0 {
            1
        } else {
            self.max_observe_stride
        }
    }

    /// Configures `max_observe_stride` (`>= 1`, where `1` forces logits observation on every token).
    #[inline(always)]
    pub const fn set_max_observe_stride(&mut self, stride: u8) {
        self.max_observe_stride = if stride == 0 { 1 } else { stride };
    }

    /// Computes effective logits observation stride (`>= 2` when `max_observe_stride > 1`, else `1`)
    /// branchlessly with `0 fdiv`, shrinking monotonically as boundary/entropy sensitivity rises.
    #[inline(always)]
    #[must_use]
    pub const fn compute_effective_observe_stride(&self, sensitivity_q7: u8) -> u8 {
        let max_s = self.effective_max_observe_stride();
        let is_dynamic = max_s > 1;
        // Map last_observed_entropy_q15 / (2 * 0.78) * 127 into [0..=64] with integer multiply-shift.
        let p_ent_raw = ((self.last_observed_entropy_q15 as u32) * 163) >> 16;
        let p_ent_q7 = if p_ent_raw > 64 { 64 } else { p_ent_raw as u8 };
        let ent_sens_q7 = sensitivity_4p1mp_q7(p_ent_q7);
        let eff_sens_q7 = if sensitivity_q7 > ent_sens_q7 {
            sensitivity_q7
        } else {
            ent_sens_q7
        };
        let span = max_s.saturating_sub(SPINE_MIN_DYNAMIC_OBSERVE_STRIDE);
        let shrink_raw = (((span as u16) * (eff_sens_q7 as u16) + 64) >> 7) as u8;
        let shrink = if shrink_raw > span { span } else { shrink_raw };
        let dyn_stride = max_s.saturating_sub(shrink);
        branchless_select_u8(is_dynamic, dyn_stride, 1)
    }

    /// Selects the logits observation mode for `next_step_index` without allocation or `fdiv`,
    /// returning `(next_observe_mode, effective_observe_stride)`. Step `0` (`next_step_index == 0`),
    /// non-greedy sampling, concrete steering steps, and `max_observe_stride <= 1` always require
    /// [`SpineLogitsObserveMode::ObserveFullLogits`].
    #[inline(always)]
    #[must_use]
    pub const fn select_next_observe_mode(
        &self,
        next_step_index: u16,
        greedy_sampling: bool,
        next_whole_token_bypass: bool,
        sensitivity_q7: u8,
    ) -> (SpineLogitsObserveMode, u8) {
        let stride = self.compute_effective_observe_stride(sensitivity_q7);
        let can_skip = (stride > 1)
            & greedy_sampling
            & (next_step_index > 0)
            & next_whole_token_bypass
            & (self.unobserved_bypass_run.saturating_add(1) < stride);
        let mode = if can_skip {
            SpineLogitsObserveMode::SkipToDeviceArgmax
        } else {
            SpineLogitsObserveMode::ObserveFullLogits
        };
        (mode, stride)
    }

    /// Arms asymmetric bypass recovery cooldown on exit instability or `Lacuna::Undefined`.
    #[inline(always)]
    pub const fn arm_bypass_cooldown(&mut self, cooldown_steps: u8) {
        let min_cd = if cooldown_steps == 0 {
            1
        } else {
            cooldown_steps
        };
        if self.bypass_cooldown_steps < min_cd {
            self.bypass_cooldown_steps = min_cd;
        }
        self.bypass_dwell_steps = 0;
        self.whole_token_bypass_active = 0;
    }

    /// Resets per-turn whole-token bypass dwell, cooldown, concrete streaks, and residency
    /// history while preserving session configuration (`baseline_pinned_compute_mask`,
    /// `guard_hold_tokens`, `max_observe_stride`, and `PRE_BYPASS_ENABLED`).
    #[inline(always)]
    pub const fn reset_turn_bypass_state(&mut self) {
        self.whole_token_bypass_active = 0;
        self.bypass_dwell_steps = 0;
        self.bypass_cooldown_steps = 0;
        self.unobserved_bypass_run = 0;
        self.concrete_run_steps = 0;
        self.calm_concrete_streak = 0;
        self.last_veto_mask = SPINE_BYPASS_VETO_NONE;
        self.control_flags &=
            !(SPINE_CONTROL_FLAG_PREV_RESIDENCY_VALID | SPINE_CONTROL_FLAG_PREV_WAS_ABSTRACT);
    }

    /// Clears bypass dwell, cooldown, and concrete streaks (preserving residency valid flags).
    #[inline(always)]
    pub const fn reset_turn_bypass_counters_only(&mut self) {
        self.whole_token_bypass_active = 0;
        self.bypass_dwell_steps = 0;
        self.bypass_cooldown_steps = 0;
        self.concrete_run_steps = 0;
        self.calm_concrete_streak = 0;
    }

    /// Returns `true` when a transient contradiction-guard compute pin may safely relax toward
    /// low-cost bypass (`guard_hold_tokens == 0` disables relaxation and holds the entire turn).
    #[inline(always)]
    #[must_use]
    pub const fn can_relax_transient_guard_pin(
        &self,
        guard_hold_tokens: u8,
        liminal_stationary: u8,
        repulsion_active: bool,
        exit_entropy_rising: bool,
        min_calm_dwell: u8,
    ) -> bool {
        let req_calm = if min_calm_dwell < 2 {
            2
        } else {
            min_calm_dwell
        };
        (guard_hold_tokens > 0)
            & !repulsion_active
            & !exit_entropy_rising
            & (liminal_stationary != 0)
            & (self.concrete_run_steps >= guard_hold_tokens)
            & (self.calm_concrete_streak >= req_calm)
    }

    /// Clears transient guard-induced cooldown lag and seeds `bypass_dwell_steps` once
    /// post-hold guard pin relaxation is authorized so bypass engages smoothly.
    #[inline(always)]
    pub const fn on_transient_guard_relaxed(&mut self, min_calm_dwell: u8) {
        let req_calm = if min_calm_dwell < 2 {
            2
        } else {
            min_calm_dwell
        };
        self.bypass_cooldown_steps = 0;
        if self.bypass_dwell_steps < req_calm {
            self.bypass_dwell_steps = req_calm;
        }
    }

    /// Evaluates whole-token steering bypass directly on `self` (`0 alloc`, `0 fdiv`) and updates
    /// `bypass_dwell_steps`, `bypass_cooldown_steps`, `whole_token_bypass_active`,
    /// `concrete_run_steps`, `calm_concrete_streak`, and `dispatch.whole_token_steering_bypass`.
    #[inline]
    pub const fn evaluate_whole_token_bypass_branchless(
        &mut self,
        obs: TokenBypassObservation,
        spine_vetoed: bool,
        min_cooldown_steps: u8,
        dispatch: &mut ComputeDispatchMask,
    ) -> bool {
        let raw_irreversible = obs.irreversible_step != 0;
        let unsafe_or_boundary = raw_irreversible
            | spine_vetoed
            | (obs.repulsion_active != 0)
            | (obs.exit_entropy_rising != 0)
            | (obs.boundary_sensitivity_q7 >= 80)
            | (obs.ego_active_boundary != 0);

        let min_cd = if min_cooldown_steps < 2 {
            2
        } else {
            min_cooldown_steps
        };
        let req_dwell = if obs.min_calm_dwell < 2 {
            2
        } else {
            obs.min_calm_dwell
        };

        let ready = if unsafe_or_boundary {
            self.bypass_dwell_steps = 0;
            if self.bypass_cooldown_steps < min_cd {
                self.bypass_cooldown_steps = min_cd;
            }
            false
        } else {
            self.bypass_cooldown_steps = self.bypass_cooldown_steps.saturating_sub(1);
            if obs.liminal_stationary == 0 {
                self.bypass_dwell_steps = 0;
                false
            } else {
                self.bypass_dwell_steps = self.bypass_dwell_steps.saturating_add(1);
                (self.bypass_cooldown_steps == 0) & (self.bypass_dwell_steps >= req_dwell)
            }
        };

        let ready_u8 = ready as u8;
        self.whole_token_bypass_active = ready_u8;
        dispatch.whole_token_steering_bypass = ready_u8;

        // Track concrete decode streaks independently of spine-vetoed `bypass_dwell_steps`
        // so transient guard pins can relax after holding the opening window.
        let is_concrete = !ready;
        self.concrete_run_steps =
            branchless_select_u8(is_concrete, self.concrete_run_steps.saturating_add(1), 0);
        let is_calm_concrete = is_concrete
            & !raw_irreversible
            & (obs.repulsion_active == 0)
            & (obs.exit_entropy_rising == 0)
            & (obs.liminal_stationary != 0);
        self.calm_concrete_streak = branchless_select_u8(
            is_calm_concrete,
            self.calm_concrete_streak.saturating_add(1),
            0,
        );
        ready
    }

    /// Physical `pinned` isolation barrier ensuring Ego modulation never mutates pinned apertures.
    #[inline(always)]
    #[must_use]
    pub const fn mask_ego_aperture_edits(
        &self,
        prev_aperture_mask: u32,
        next_aperture_mask: u32,
    ) -> u32 {
        let prev_cut = BitCut32::split(prev_aperture_mask, self.pinned_aperture_mask);
        let next_cut = BitCut32::split(next_aperture_mask, self.pinned_aperture_mask);
        next_cut.outside | prev_cut.inside
    }

    /// Primitive `pinned` isolation helper returning `(full_lock, effective_slot, effective_modes)`
    /// without depending on upper-layer `GatewayFrame`.
    #[inline(always)]
    #[must_use]
    pub fn isolate_slot_and_modes_branchless(
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

    /// Pins compute slots in `pinned_compute_mask` so Fast-Loop bypass cannot skip them.
    #[inline(always)]
    pub const fn pin_compute_slots(&mut self, slot_mask: u64) {
        self.pinned_compute_mask |= slot_mask;
    }

    /// Unpins compute slots from `pinned_compute_mask`.
    #[inline(always)]
    pub const fn unpin_compute_slots(&mut self, slot_mask: u64) {
        self.pinned_compute_mask &= !slot_mask;
    }

    /// Clears all bits in `pinned_compute_mask`.
    #[inline(always)]
    pub const fn clear_pinned_compute_slots(&mut self) {
        self.pinned_compute_mask = 0;
    }

    /// Evaluates branchless whole-token bypass veto flags (`SPINE_BYPASS_VETO_*`).
    ///
    /// Explicitly excludes `unobserved_locked` so OOD (`Lacuna::Unobserved`) does not trigger
    /// guard/repulsion bypass vetoes.
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_whole_token_bypass_veto_branchless(
        &self,
        guard_active: bool,
        repulsion_active: bool,
    ) -> u8 {
        let g = (guard_active as u8) * SPINE_BYPASS_VETO_GUARD_ACTIVE;
        let r = (repulsion_active as u8) * SPINE_BYPASS_VETO_REPULSION_ACTIVE;
        let pinned = (self.irreversible_step_lock != 0) | (self.pinned_compute_mask != 0);
        let p = (pinned as u8) * SPINE_BYPASS_VETO_PINNED_LOCK;
        g | r | p
    }

    /// Evaluates branchless whole-token bypass veto flags (`SPINE_BYPASS_VETO_*`) against an
    /// anonymous compute pool's `reversible_mask`.
    ///
    /// Unlike [`Self::evaluate_whole_token_bypass_veto_branchless`], stationary boundary pins
    /// outside `reversible_mask` (such as permanent head/tail anchors) protect their own slots
    /// from layer bypass without vetoing calm whole-token bypass; only `irreversible_step_lock`
    /// or dynamic pins overlapping `reversible_mask` raise `SPINE_BYPASS_VETO_PINNED_LOCK`.
    #[inline(always)]
    #[must_use]
    pub const fn evaluate_pool_bypass_veto_branchless(
        &self,
        reversible_mask: u64,
        guard_active: bool,
        repulsion_active: bool,
    ) -> u8 {
        let g = (guard_active as u8) * SPINE_BYPASS_VETO_GUARD_ACTIVE;
        let r = (repulsion_active as u8) * SPINE_BYPASS_VETO_REPULSION_ACTIVE;
        let dynamic_pin_overlap =
            BitCut64::split(self.pinned_compute_mask, reversible_mask).inside != 0;
        let pinned = (self.irreversible_step_lock != 0) | dynamic_pin_overlap;
        let p = (pinned as u8) * SPINE_BYPASS_VETO_PINNED_LOCK;
        g | r | p
    }

    /// Returns `true` only when `candidate_bypass` is requested and no spine bypass veto is set.
    #[inline(always)]
    #[must_use]
    pub const fn should_bypass_whole_token_branchless(
        &self,
        candidate_bypass: bool,
        guard_active: bool,
        repulsion_active: bool,
    ) -> bool {
        candidate_bypass
            & (self.evaluate_whole_token_bypass_veto_branchless(guard_active, repulsion_active)
                == SPINE_BYPASS_VETO_NONE)
    }

    /// Masks a candidate layer bypass bitmask against `pinned_compute_mask` and `veto_mask`
    /// without branches: returns `0` when `veto_mask != 0`, else `raw_bypass_mask & !pinned`.
    #[inline(always)]
    #[must_use]
    pub const fn mask_layer_bypass_with_pinned(&self, raw_bypass_mask: u64, veto_mask: u8) -> u64 {
        let allow_mask = ((veto_mask == SPINE_BYPASS_VETO_NONE) as u64).wrapping_neg();
        BitCut64::split(raw_bypass_mask, self.pinned_compute_mask).outside & allow_mask
    }

    /// Evaluates an I-Frame / P-Frame transition branchlessly (`0 alloc`, `0 fdiv`), returning
    /// `(directive, allowed_granule_mask)`.
    ///
    /// Priority order (highest to lowest):
    /// 1. [`Lacuna::Undefined`] (`>= 3`) -> [`FrameDirective::RejectUndefined`] (`allowed = 0`).
    /// 2. Spine lock (`irreversible_step_lock`, `SPINE_TRANSITION_FLAG_IRREVERSIBLE`, or `unobserved_locked`)
    ///    -> [`FrameDirective::PassThrough`] (`allowed = 0`).
    /// 3. Defined observed re-anchor (`!Unobserved` and (`max_gop_steps` expired, `NEW_BASIN_SPAWNED`,
    ///    `GUARD_REPULSION`, [`Lacuna::Exterior`], or `phase_delta_q7 >= reanchor_delta_thresh_q7`))
    ///    -> [`FrameDirective::ReAnchorKeyframe`] (`allowed = non_pinned`).
    /// 4. [`Lacuna::Unobserved`] (`0`) or (`SelfRef` with `STATIONARY_BYPASS_READY`)
    ///    -> [`FrameDirective::Skip`] (`allowed = 0`; prevents OOD queries from spawning spurious keyframes).
    /// 5. Otherwise -> [`FrameDirective::EmitDelta`] (`allowed = non_pinned`).
    #[inline]
    #[must_use]
    pub const fn evaluate_frame_transition_branchless(
        &self,
        input: SpineFrameTransitionInput,
    ) -> (FrameDirective, u8) {
        let is_unobserved = input.lacuna_code == Lacuna::Unobserved.code();
        let is_self_ref = input.lacuna_code == Lacuna::SelfRef.code();
        let is_exterior = input.lacuna_code == Lacuna::Exterior.code();
        let is_undefined = input.lacuna_code >= Lacuna::Undefined.code();

        let irreversible = (self.irreversible_step_lock != 0)
            | ((input.flags & SPINE_TRANSITION_FLAG_IRREVERSIBLE) != 0);
        let ood_locked = self.unobserved_locked != 0;
        let guard_repulsion = (input.flags & SPINE_TRANSITION_FLAG_GUARD_REPULSION) != 0;
        let new_basin = (input.flags & SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED) != 0;
        let stationary_ready = (input.flags & SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY) != 0;

        let gop_expired =
            (input.max_gop_steps > 0) & (input.steps_since_keyframe >= input.max_gop_steps);
        let delta_exceeded = (!is_unobserved)
            & (input.reanchor_delta_thresh_q7 > 0)
            & (input.phase_delta_q7 >= input.reanchor_delta_thresh_q7);

        let want_reanchor = (!is_unobserved)
            & (gop_expired | new_basin | guard_repulsion | is_exterior | delta_exceeded);
        let want_skip = is_unobserved | (is_self_ref & stationary_ready);

        let mut dir = FrameDirective::EmitDelta as u8;
        dir = branchless_select_u8(want_skip, FrameDirective::Skip as u8, dir);
        dir = branchless_select_u8(want_reanchor, FrameDirective::ReAnchorKeyframe as u8, dir);
        dir = branchless_select_u8(
            irreversible | ood_locked,
            FrameDirective::PassThrough as u8,
            dir,
        );
        dir = branchless_select_u8(is_undefined, FrameDirective::RejectUndefined as u8, dir);

        let non_pinned =
            BitCut8::split(input.candidate_granule_mask, self.pinned_slot_mask).outside;
        let allow_mod = (dir == FrameDirective::EmitDelta as u8)
            | (dir == FrameDirective::ReAnchorKeyframe as u8);
        let allowed_granule_mask = branchless_select_u8(allow_mod, non_pinned, 0);

        let directive = match dir {
            0 => FrameDirective::Skip,
            1 => FrameDirective::PassThrough,
            2 => FrameDirective::EmitDelta,
            3 => FrameDirective::ReAnchorKeyframe,
            _ => FrameDirective::RejectUndefined,
        };
        (directive, allowed_granule_mask)
    }
}

// SAFETY: `SystemSpineState` is `#[repr(C, align(64))]`, exact size 64 bytes with zero implicit padding.
unsafe impl Pod for SystemSpineState {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_frame_transition_branchless_enforces_priority_and_ood_isolation() {
        let mut spine = SystemSpineState::new();
        let baseline_mask = 0xFF_u8;
        let expected_non_pinned = baseline_mask & !spine.pinned_slot_mask;

        // 1. Nominal SelfRef -> EmitDelta with pinned granules masked out.
        let nominal = SpineFrameTransitionInput {
            lacuna_code: Lacuna::SelfRef.code(),
            phase_delta_q7: 8,
            reanchor_delta_thresh_q7: 32,
            steps_since_keyframe: 2,
            max_gop_steps: 16,
            candidate_granule_mask: baseline_mask,
            flags: 0,
            reserved: 0,
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(nominal),
            (FrameDirective::EmitDelta, expected_non_pinned)
        );
        assert_eq!(expected_non_pinned & spine.pinned_slot_mask, 0);

        // 2. Stationary shortcut on SelfRef -> Skip with 0 mask.
        let stationary = SpineFrameTransitionInput {
            flags: SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
            ..nominal
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(stationary),
            (FrameDirective::Skip, 0)
        );

        // 3. ReAnchorKeyframe overrides stationary skip on GOP expiry, phase delta, new basin,
        //    guard repulsion, or Lacuna::Exterior.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                steps_since_keyframe: 16,
                ..stationary
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                phase_delta_q7: 32,
                ..stationary
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                flags: SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED
                    | SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                flags: SPINE_TRANSITION_FLAG_GUARD_REPULSION,
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Exterior.code(),
                ..nominal
            }),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );

        // 4. Unobserved OOD zero-aperture MUST return Skip and never spawn spurious keyframes
        //    even if delta/GOP/basin flags are set.
        let ood_noisy = SpineFrameTransitionInput {
            lacuna_code: Lacuna::Unobserved.code(),
            phase_delta_q7: 120,
            steps_since_keyframe: 20,
            flags: SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED | SPINE_TRANSITION_FLAG_GUARD_REPULSION,
            ..nominal
        };
        assert_eq!(
            spine.evaluate_frame_transition_branchless(ood_noisy),
            (FrameDirective::Skip, 0)
        );

        // 5. Irreversible flag or spine lock forces PassThrough above ReAnchorKeyframe/Skip.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Exterior.code(),
                flags: SPINE_TRANSITION_FLAG_IRREVERSIBLE,
                ..nominal
            }),
            (FrameDirective::PassThrough, 0)
        );
        spine.unobserved_locked = 1;
        assert_eq!(
            spine.evaluate_frame_transition_branchless(nominal),
            (FrameDirective::PassThrough, 0)
        );

        // 6. Lacuna::Undefined dominates everything -> RejectUndefined.
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: Lacuna::Undefined.code(),
                ..nominal
            }),
            (FrameDirective::RejectUndefined, 0)
        );
        assert_eq!(
            spine.evaluate_frame_transition_branchless(SpineFrameTransitionInput {
                lacuna_code: 255,
                ..nominal
            }),
            (FrameDirective::RejectUndefined, 0)
        );
    }

    #[test]
    fn test_spine_frame_transition_input_from_lacuna_partition() {
        let mut spine = SystemSpineState::new();
        spine.pinned_slot_mask = 0b0000_0100;
        let expected_non_pinned = 0b0011_1100_u8 & !spine.pinned_slot_mask;

        // 1. Unobserved (S \cup O complement): zero-aperture isolation -> Skip regardless of delta/GOP.
        let unobserved_part = crate::liminal::LacunaPartition8 {
            unobserved_mask: 0xFF,
            ..Default::default()
        };
        let unobserved_in = SpineFrameTransitionInput::from_lacuna_partition(
            &unobserved_part,
            120,
            16,
            24,
            16,
            SPINE_TRANSITION_FLAG_NEW_BASIN_SPAWNED | SPINE_TRANSITION_FLAG_GUARD_REPULSION,
        );
        assert_eq!(unobserved_in.lacuna_code, Lacuna::Unobserved.code());
        assert_eq!(unobserved_in.candidate_granule_mask, 0);
        assert_eq!(
            spine.evaluate_frame_transition_branchless(unobserved_in),
            (FrameDirective::Skip, 0)
        );

        // 2. SelfRef stationary bypass vs nominal delta emission.
        let self_ref_part = crate::liminal::LacunaPartition8 {
            self_ref_mask: 0b0011_1100,
            unobserved_mask: 0b1100_0011,
            ..Default::default()
        };
        let self_ref_stationary = SpineFrameTransitionInput::from_lacuna_partition(
            &self_ref_part,
            2,
            16,
            1,
            16,
            SPINE_TRANSITION_FLAG_STATIONARY_BYPASS_READY,
        );
        assert_eq!(self_ref_stationary.lacuna_code, Lacuna::SelfRef.code());
        assert_eq!(self_ref_stationary.candidate_granule_mask, 0b0011_1100);
        assert_eq!(
            spine.evaluate_frame_transition_branchless(self_ref_stationary),
            (FrameDirective::Skip, 0)
        );
        let self_ref_delta =
            SpineFrameTransitionInput::from_lacuna_partition(&self_ref_part, 2, 16, 1, 16, 0);
        assert_eq!(
            spine.evaluate_frame_transition_branchless(self_ref_delta),
            (FrameDirective::EmitDelta, expected_non_pinned)
        );

        // 3. Exterior dominates SelfRef -> sandbox delta / ReAnchorKeyframe.
        let exterior_part = crate::liminal::LacunaPartition8 {
            self_ref_mask: 0b0000_1100,
            exterior_mask: 0b0011_0000,
            unobserved_mask: 0b1100_0011,
            ..Default::default()
        };
        let exterior_in =
            SpineFrameTransitionInput::from_lacuna_partition(&exterior_part, 2, 16, 1, 16, 0);
        assert_eq!(exterior_in.lacuna_code, Lacuna::Exterior.code());
        assert_eq!(exterior_in.candidate_granule_mask, 0b0011_1100);
        assert_eq!(
            spine.evaluate_frame_transition_branchless(exterior_in),
            (FrameDirective::ReAnchorKeyframe, expected_non_pinned)
        );

        // 4. Undefined dominates all regimes -> RejectUndefined fault isolation.
        let undefined_part = crate::liminal::LacunaPartition8 {
            self_ref_mask: 0b0000_0100,
            exterior_mask: 0b0000_1000,
            undefined_mask: 0b0001_0000,
            unobserved_mask: 0b1110_0011,
            ..Default::default()
        };
        let undefined_in =
            SpineFrameTransitionInput::from_lacuna_partition(&undefined_part, 2, 16, 1, 16, 0);
        assert_eq!(undefined_in.lacuna_code, Lacuna::Undefined.code());
        assert_eq!(
            spine.evaluate_frame_transition_branchless(undefined_in),
            (FrameDirective::RejectUndefined, 0)
        );
    }

    #[test]
    fn spine_decode_input_encapsulates_lacuna_thresholds_and_sentinel() {
        let p = SystemPressureReading::default();
        let calm = SpineDecodeInput::from_observed_entropy_f32(p, 0, 0.50, 12, false, true);
        assert_eq!(calm.exit_entropy_lacuna(), Lacuna::SelfRef);
        assert!(
            calm.observed_exit_entropy_q15() > 16000 && calm.observed_exit_entropy_q15() < 16600
        );
        assert!(calm.greedy_sampling());
        assert!(!calm.repulsion_active());
        assert!(calm.pre_bypass_enabled());
        assert_eq!(calm.exit_reward_override_q7(), SPINE_SENTINEL_AUTO_U8);

        // Zero-initialized (`memset(0)`) SpineDecodeInput must leave exit_reward_override_q7 automatic (`0xFF`).
        let zero_init = SpineDecodeInput::default();
        assert_eq!(zero_init.flags, 0);
        assert_eq!(zero_init.exit_reward_override_q7, 0);
        assert_eq!(
            SpineDecodeSignal::exit_reward_override_q7(&zero_init),
            SPINE_SENTINEL_AUTO_U8
        );

        // Even non-zero raw `exit_reward_override_q7` stays ignored until `SPINE_DECODE_FLAG_REWARD_OVERRIDE` is set.
        let mut unflagged = zero_init;
        unflagged.exit_reward_override_q7 = 42;
        assert_eq!(
            SpineDecodeSignal::exit_reward_override_q7(&unflagged),
            SPINE_SENTINEL_AUTO_U8
        );

        // Flagged reward override honors `0..=127` and clamps out-of-range `> 127` values to `127`.
        let ov_zero = calm.with_exit_reward_override_q7(0);
        assert_ne!(ov_zero.flags & SPINE_DECODE_FLAG_REWARD_OVERRIDE, 0);
        assert_eq!(SpineDecodeSignal::exit_reward_override_q7(&ov_zero), 0);
        let ov_clamped = calm.with_exit_reward_override_q7(250);
        assert_eq!(SpineDecodeSignal::exit_reward_override_q7(&ov_clamped), 127);
        let ov_cleared = ov_clamped.without_exit_reward_override();
        assert_eq!(
            SpineDecodeSignal::exit_reward_override_q7(&ov_cleared),
            SPINE_SENTINEL_AUTO_U8
        );

        let boundary = SpineDecodeInput::from_observed_entropy_f32(p, 1, 0.78, 12, false, true);
        assert_eq!(boundary.exit_entropy_lacuna(), Lacuna::Exterior);

        let unobs = SpineDecodeInput::unobserved(p, 2, 16, true, true).with_pre_bypass(false);
        assert_eq!(unobs.exit_entropy_lacuna(), Lacuna::Unobserved);
        assert_eq!(unobs.observed_exit_entropy_q15(), 0);
        assert!(unobs.repulsion_active());
        assert!(!unobs.pre_bypass_enabled());

        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let undef = SpineDecodeInput::from_observed_entropy_f32(p, 3, bad, 0, false, true);
            assert_eq!(undef.exit_entropy_lacuna(), Lacuna::Undefined);
            assert_eq!(undef.observed_exit_entropy_q15(), 0);
        }
    }

    #[test]
    fn spine_observe_stride_shrinks_monotonically_and_guards_baseline_pins() {
        let mut spine = SystemSpineState::new();
        assert_eq!(spine.effective_max_observe_stride(), 1);
        assert_eq!(spine.compute_effective_observe_stride(0), 1);

        // Zero-defensive fallback: even all-zero spine operates at stride 1.
        let zeroed = SystemSpineState {
            max_observe_stride: 0,
            ..SystemSpineState::new()
        };
        assert_eq!(zeroed.effective_max_observe_stride(), 1);
        assert_eq!(
            zeroed.select_next_observe_mode(3, true, true, 0),
            (SpineLogitsObserveMode::ObserveFullLogits, 1)
        );

        spine.set_max_observe_stride(8);
        let mut prev_stride = u8::MAX;
        for &(sens_q7, ent_f32) in &[(0u8, 0.0f32), (24, 0.25), (48, 0.50), (64, 0.78)] {
            spine.last_observed_entropy_q15 = ((ent_f32 * f32::from(SPINE_ENTROPY_Q15_ONE) + 0.5)
                as u16)
                .min(SPINE_ENTROPY_Q15_ONE);
            let s = spine.compute_effective_observe_stride(sens_q7);
            assert!((2..=8).contains(&s));
            assert!(s <= prev_stride);
            prev_stride = s;
        }
        spine.last_observed_entropy_q15 = 0;
        assert_eq!(spine.compute_effective_observe_stride(0), 8);
        spine.last_observed_entropy_q15 = SPINE_ENTROPY_CALM_THRESH_Q15;
        assert_eq!(spine.compute_effective_observe_stride(126), 2);

        // Step 0, non-greedy, and non-bypass always observe full logits.
        spine.last_observed_entropy_q15 = 0;
        assert_eq!(
            spine.select_next_observe_mode(0, true, true, 0),
            (SpineLogitsObserveMode::ObserveFullLogits, 8)
        );
        assert_eq!(
            spine.select_next_observe_mode(1, false, true, 0),
            (SpineLogitsObserveMode::ObserveFullLogits, 8)
        );
        assert_eq!(
            spine.select_next_observe_mode(1, true, false, 0),
            (SpineLogitsObserveMode::ObserveFullLogits, 8)
        );
        assert_eq!(
            spine.select_next_observe_mode(1, true, true, 0),
            (SpineLogitsObserveMode::SkipToDeviceArgmax, 8)
        );

        // Transient guard pin does not pollute baseline_pinned_compute_mask.
        spine.apply_host_pin_floor(0b0000_1010);
        spine.arm_transient_guard_pin(0xFFFF_FFFF);
        assert_eq!(spine.baseline_pinned_compute_mask, 0b0000_1010);
        assert_eq!(spine.pinned_compute_mask, 0xFFFF_FFFF);
        spine.restore_baseline_pin_mask();
        assert_eq!(
            spine.pinned_compute_mask,
            0b0000_1010 | HOST_IMMUTABLE_PINNED_COMPUTE_MASK
        );
    }
}
