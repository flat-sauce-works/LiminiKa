// name: src/core/src/stigmergy/circuit.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-scale swarm hierarchy, [`CognitiveTrajectoryLedger`], and
//! [`CognitiveCircuitBankOps`] / [`CognitiveCircuitBank`] phase-word abstractions.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]

use crate::foundation::{
    branchless_clamp_f32, swar_q7_abs_u8_x8, swar_q7_circular_diff_x8, words_to_q7_array_64, Pod,
    RawPhaseProjectable, SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{gcso_q7_t, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES};
use crate::stigmergy::action_hub::StigmergicThresholds;
use crate::stigmergy::ledger::gcso_ledger_slot_t;
use crate::stigmergy::{LEDGER_FLAG_COGNITIVE_TRAJECTORY, LEDGER_HISTORY_LEN};

/// Multi-scale granularity tier within the self-similar GCSO hierarchy.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum StigmergicScaleGranularity {
    /// Scale 0: 64-bit tagged pointer and single 8-head `u64` SWAR register.
    PointerBit = 0,
    /// Scale 1: 128-byte pointer trail (`8` granules x `8` heads) with local inhibition.
    CellTrail = 1,
    /// Scale 2: Type-level blanket composition and 128-byte `GatewayFrame` interference.
    #[default]
    TypeGateway = 2,
    /// Scale 3..N: Macro-level multi-hop pointer ring and cross-stratum coalition cascade.
    MacroCoalition = 3,
}

impl StigmergicScaleGranularity {
    /// Converts a raw 2-bit integer (`0..=3`) into a [`StigmergicScaleGranularity`].
    #[inline(always)]
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x03 {
            0 => Self::PointerBit,
            1 => Self::CellTrail,
            2 => Self::TypeGateway,
            _ => Self::MacroCoalition,
        }
    }

    /// Maps `(layer_index, total_layers)` monotonically across the 4 structural scales,
    /// progressing from coarse macro context (`MacroCoalition`) to fine residual (`PointerBit`).
    #[inline(always)]
    #[must_use]
    pub const fn from_layer_ratio(layer_index: usize, total_layers: usize) -> Self {
        let safe_total = if total_layers == 0 { 1 } else { total_layers };
        let clamped_idx = if layer_index >= safe_total {
            safe_total - 1
        } else {
            layer_index
        };
        let q4 = (clamped_idx * 4) / safe_total;
        match q4 {
            0 => Self::MacroCoalition,
            1 => Self::TypeGateway,
            2 => Self::CellTrail,
            _ => Self::PointerBit,
        }
    }

    /// Derives the scale-adjusted Micro-Swarm quorum threshold (`3..=7` out of 8 cells).
    #[inline(always)]
    #[must_use]
    pub const fn swarm_quorum_requirement(self, base_quorum: u8) -> u8 {
        let safe_base = if base_quorum == 0 {
            1
        } else if base_quorum > 8 {
            8
        } else {
            base_quorum
        };
        let adj = match self {
            Self::PointerBit => safe_base.saturating_sub(2),
            Self::CellTrail => safe_base.saturating_sub(1),
            Self::TypeGateway => safe_base,
            Self::MacroCoalition => {
                let v = safe_base.saturating_add(1);
                if v > 7 {
                    7
                } else {
                    v
                }
            }
        };
        if adj < 2 {
            2
        } else {
            adj
        }
    }

    /// Derives the scale-adjusted Q7 circular phase tolerance (`4..=32`).
    #[inline(always)]
    #[must_use]
    pub const fn swarm_tolerance_q7(self, base_tol_q7: u8) -> u8 {
        let safe = if base_tol_q7 < 4 { 4 } else { base_tol_q7 };
        let scaled = match self {
            Self::PointerBit => safe,
            Self::CellTrail => safe.saturating_add(4),
            Self::TypeGateway => safe.saturating_add(8),
            Self::MacroCoalition => safe.saturating_add(14),
        };
        if scaled > 40 {
            40
        } else {
            scaled
        }
    }

    /// Derives the stigmergic peeling strength in Q0.7 (`24..=128`) from this scale and `layer_index / total_layers`.
    #[inline(always)]
    #[must_use]
    pub const fn layer_peel_gain_q7(self, layer_index: usize, total_layers: usize) -> u8 {
        let safe_total = if total_layers == 0 { 1 } else { total_layers };
        let base_q7: u8 = match self {
            Self::PointerBit => 40,
            Self::CellTrail => 64,
            Self::TypeGateway => 96,
            Self::MacroCoalition => 112,
        };
        let clamped_idx = if layer_index > 8 { 8 } else { layer_index } as u8;
        let depth_decay = (clamped_idx.saturating_mul(96)) / (safe_total as u8);
        let out = base_q7.saturating_sub(depth_decay >> 2);
        if out < 24 {
            24
        } else {
            out
        }
    }
}

/// Functional cognitive trigger recorded along a multi-scale reasoning trajectory.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum CognitiveOpKind {
    /// Low-interference baseline observation.
    #[default]
    Observe = 0,
    /// Orthogonal residual peeling and coalition decomposition.
    DecomposePeel = 1,
    /// Coherent cross-basin synthesis and integration.
    SynthesizeFuse = 2,
    /// Contrastive Guard audit and anti-phase repulsion check.
    RepulseAudit = 3,
    /// Inter-basin perspective shift across a liminal boundary.
    BridgeShift = 4,
    /// RoPE phase steering and alignment correction.
    SteerAlign = 5,
    /// Specialized sub-head (`Fact / Logic / Explore`) routing.
    RouteSubHead = 6,
    /// Macro attractor crystallization and structural commitment.
    CrystallizeCommit = 7,
}

impl CognitiveOpKind {
    /// Converts a raw 3-bit identifier (`0..=7`) into a [`CognitiveOpKind`].
    #[inline(always)]
    #[must_use]
    pub const fn from_u8(val: u8) -> Self {
        match val & 0x07 {
            0 => Self::Observe,
            1 => Self::DecomposePeel,
            2 => Self::SynthesizeFuse,
            3 => Self::RepulseAudit,
            4 => Self::BridgeShift,
            5 => Self::SteerAlign,
            6 => Self::RouteSubHead,
            _ => Self::CrystallizeCommit,
        }
    }
}

/// 1-byte compact encoding of a single multi-scale cognitive action step (`[Scale(2b) | Op(3b) | Intensity(3b)]`).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CognitiveStepOp(
    /// Raw packed 8-bit cognitive step descriptor (`0` indicates an unrecorded slot).
    pub u8,
);

// SAFETY: `CognitiveStepOp` is a 1-byte `#[repr(transparent)]` wrapper around `u8`.
unsafe impl Pod for CognitiveStepOp {}

impl CognitiveStepOp {
    /// Packs `granularity` (2 bits), `op` (3 bits), and `intensity_0_to_7` (3 bits) into 1 byte.
    #[inline(always)]
    #[must_use]
    pub const fn pack(
        granularity: StigmergicScaleGranularity,
        op: CognitiveOpKind,
        intensity_0_to_7: u8,
    ) -> Self {
        let g = ((granularity as u8) & 0x03) << 6;
        let o = ((op as u8) & 0x07) << 3;
        let i = intensity_0_to_7 & 0x07;
        Self(g | o | i)
    }

    /// Returns `true` when this slot holds a recorded cognitive step (`raw != 0`).
    #[inline(always)]
    #[must_use]
    pub const fn is_active(self) -> bool {
        self.0 != 0
    }

    /// Unpacks the 2-bit [`StigmergicScaleGranularity`].
    #[inline(always)]
    #[must_use]
    pub const fn granularity(self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::from_u8(self.0 >> 6)
    }

    /// Unpacks the 3-bit [`CognitiveOpKind`].
    #[inline(always)]
    #[must_use]
    pub const fn op(self) -> CognitiveOpKind {
        CognitiveOpKind::from_u8(self.0 >> 3)
    }

    /// Unpacks the 3-bit quantized firing intensity (`0..=7`).
    #[inline(always)]
    #[must_use]
    pub const fn intensity(self) -> u8 {
        self.0 & 0x07
    }

    /// Projects this cognitive step into a non-wrapping signed Q7 phase contribution (`[-120, +120]`).
    #[inline(always)]
    #[must_use]
    pub const fn to_signed_q7(self) -> gcso_q7_t {
        if self.0 == 0 {
            return 0;
        }
        let mag =
            (16i16) + ((self.intensity() as i16) * 8) + (((self.granularity() as u8) as i16) * 12);
        let clamped = if mag > 120 { 120 } else { mag };
        match self.op() {
            CognitiveOpKind::RepulseAudit | CognitiveOpKind::DecomposePeel => -(clamped as i8),
            _ => clamped as i8,
        }
    }
}

/// 8-byte C-ABI register-resident activation output produced by evaluating a stigmergic
/// cognitive phase [`CognitiveTrajectoryLedger`] circuit at a given scale.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CognitiveCircuitActivation {
    /// Propensity in Q0.7 (`0..=128`) to decompose / peel orthogonal residuals.
    pub peel_gate_q7: u8,
    /// Propensity in Q0.7 (`0..=128`) to fuse / integrate coherent basins.
    pub fuse_gate_q7: u8,
    /// Contrastive Guard sensitivity in Q0.7 (`0..=128`).
    pub repulse_sensitivity_q7: u8,
    /// Selected [`CognitiveOpKind`] discriminant (`0..=7`).
    pub selected_op: u8,
    /// Preferred next [`StigmergicScaleGranularity`] discriminant (`0..=3`) for perspective transition.
    pub preferred_next_scale: u8,
    /// Firing intensity in Q0.7 (`0..=128`).
    pub intensity_q7: u8,
    /// Signed SWAR alignment (`-127..=127`) between the circuit bias and input field at this scale.
    pub circuit_resonance_q7: gcso_q7_t,
    /// Active recorded step count observed inside the trajectory ledger (`0..=8`).
    pub trajectory_depth: u8,
}

// SAFETY: `CognitiveCircuitActivation` is 8-byte `#[repr(C, align(8))]` of `u8`/`i8` fields with zero padding.
unsafe impl Pod for CognitiveCircuitActivation {}

impl CognitiveCircuitActivation {
    /// Returns the typed [`CognitiveOpKind`] selected by this circuit activation.
    #[inline(always)]
    #[must_use]
    pub const fn op_kind(&self) -> CognitiveOpKind {
        CognitiveOpKind::from_u8(self.selected_op)
    }

    /// Returns the typed preferred next [`StigmergicScaleGranularity`].
    #[inline(always)]
    #[must_use]
    pub const fn next_scale(&self) -> StigmergicScaleGranularity {
        StigmergicScaleGranularity::from_u8(self.preferred_next_scale)
    }
}

/// 8-byte (`#[repr(C, align(8))]`) chronological trajectory ledger encoding the last 8
/// multi-scale cognitive operations (`[Scale | Op | Intensity]`) inside a single `u64` register.
///
/// Mirrors [`gcso_ledger_slot_t::telemetry_mini_ledger`] 1:1 so observed scale transitions and
/// reasoning habits act directly as reusable persona circuit parameters via
/// `ManifoldGatewaySignal` and `StigmergicScaleLayer`.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct CognitiveTrajectoryLedger {
    /// Chronological byte slots where index `0` is the most recent [`CognitiveStepOp`].
    pub steps: [u8; LEDGER_HISTORY_LEN],
}

// SAFETY: `CognitiveTrajectoryLedger` is `#[repr(C, align(8))]` wrapping `[u8; 8]` with zero padding.
unsafe impl Pod for CognitiveTrajectoryLedger {}

#[inline(always)]
#[must_use]
pub(crate) const fn swar_mean_u8_le128_x8(w: u64) -> u8 {
    let even = w & 0x00FF_00FF_00FF_00FF;
    let odd = (w >> 8) & 0x00FF_00FF_00FF_00FF;
    let sum16 = even.wrapping_add(odd);
    let total = (sum16.wrapping_mul(0x0001_0001_0001_0001) >> 48) as u16;
    ((total + 4) >> 3) as u8
}

impl CognitiveTrajectoryLedger {
    /// Creates an empty [`CognitiveTrajectoryLedger`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            steps: [0u8; LEDGER_HISTORY_LEN],
        }
    }

    /// Constructs a [`CognitiveTrajectoryLedger`] from a packed 64-bit word.
    #[inline(always)]
    #[must_use]
    pub const fn from_u64(raw: u64) -> Self {
        Self {
            steps: raw.to_le_bytes(),
        }
    }

    /// Converts the 8-byte chronological trajectory into a single `u64` register word.
    #[inline(always)]
    #[must_use]
    pub const fn to_u64(self) -> u64 {
        u64::from_le_bytes(self.steps)
    }

    /// Loads the cognitive trajectory directly from `slot.telemetry_mini_ledger`.
    #[inline(always)]
    #[must_use]
    pub const fn from_ledger_slot(slot: &gcso_ledger_slot_t) -> Self {
        Self {
            steps: slot.telemetry_mini_ledger,
        }
    }

    /// Stores this trajectory back into `slot.telemetry_mini_ledger` and sets
    /// [`LEDGER_FLAG_COGNITIVE_TRAJECTORY`] in `slot.auto_tune_flags`.
    #[inline(always)]
    pub const fn write_to_ledger_slot(&self, slot: &mut gcso_ledger_slot_t) {
        slot.telemetry_mini_ledger = self.steps;
        slot.auto_tune_flags |= LEDGER_FLAG_COGNITIVE_TRAJECTORY;
    }

    /// Shifts chronological history by one byte and pushes `step` into slot `0`.
    #[inline(always)]
    pub fn push_step(&mut self, step: CognitiveStepOp) {
        let next = (self.to_u64() << 8) | u64::from(step.0);
        *self = Self::from_u64(next);
    }

    /// Returns the most recent recorded [`CognitiveStepOp`], if non-zero.
    #[inline(always)]
    #[must_use]
    pub const fn latest_step(&self) -> Option<CognitiveStepOp> {
        if self.steps[0] == 0 {
            None
        } else {
            Some(CognitiveStepOp(self.steps[0]))
        }
    }

    /// Returns the `age`-th newest recorded [`CognitiveStepOp`] (`0..8`).
    #[inline(always)]
    #[must_use]
    pub const fn step_at(&self, age: usize) -> CognitiveStepOp {
        CognitiveStepOp(self.steps[age & 7])
    }

    /// Returns a constant-time histogram of [`CognitiveOpKind`] counts (`[u8; 8]`) and active step count.
    #[inline(always)]
    #[must_use]
    pub const fn op_counts(&self) -> ([u8; 8], u8) {
        let mut counts = [0u8; 8];
        let mut active = 0u8;
        let mut i = 0usize;
        while i < LEDGER_HISTORY_LEN {
            let s = CognitiveStepOp(self.steps[i]);
            if s.is_active() {
                let op = (s.0 >> 3) as usize & 0x07;
                counts[op] += 1;
                active += 1;
            }
            i += 1;
        }
        (counts, active)
    }

    /// Returns the count of recorded steps at each [`StigmergicScaleGranularity`] level (`0..4`).
    #[inline(always)]
    #[must_use]
    pub const fn scale_counts(&self) -> [u8; 4] {
        let mut counts = [0u8; 4];
        let mut i = 0usize;
        while i < LEDGER_HISTORY_LEN {
            let s = CognitiveStepOp(self.steps[i]);
            if s.is_active() {
                counts[(s.granularity() as usize) & 3] += 1;
            }
            i += 1;
        }
        counts
    }

    /// Lifts the 8 chronological steps into an 8-cell x 8-head $SO(2)^{64}$ SWAR phase field
    /// with zero heap allocations.
    #[inline(always)]
    #[must_use]
    pub const fn to_phase_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        let mut words = [0u64; NUM_STIGMERGIC_GRANULES];
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            let s = CognitiveStepOp(self.steps[c]);
            let base_q7 = s.to_signed_q7();
            let b0 = (base_q7 as u8) as u64;
            // Spread each step's polarity across the 8 heads of cell `c` weighted by scale index.
            let splat = b0.wrapping_mul(SWAR_SPLAT_BYTE_64);
            let sign_mask = ((s.0 != 0) as u64).wrapping_neg();
            words[c] = splat & sign_mask;
            c += 1;
        }
        words
    }

    /// Computes branchless temporal synaptic transition statistics across adjacent recorded
    /// steps (`prev -> curr`): `(bottom_up_momentum, top_down_peel_momentum, repulse_streak, net_scale_velocity)`.
    #[inline(always)]
    #[must_use]
    pub const fn synaptic_transition_stats(&self) -> (u8, u8, u8, i8) {
        let mut bottom_up = 0u16;
        let mut top_down_peel = 0u16;
        let mut repulse_streak = 0u16;
        let mut scale_vel = 0i16;

        let latest = CognitiveStepOp(self.steps[0]);
        if latest.is_active() && matches!(latest.op(), CognitiveOpKind::RepulseAudit) {
            repulse_streak += 12;
        }

        let mut i = 0usize;
        while i + 1 < LEDGER_HISTORY_LEN {
            let curr = CognitiveStepOp(self.steps[i]);
            let prev = CognitiveStepOp(self.steps[i + 1]);
            if curr.is_active() && prev.is_active() {
                let curr_s = curr.granularity() as i16;
                let prev_s = prev.granularity() as i16;
                let ds = curr_s - prev_s;
                scale_vel += ds;
                if ds > 0 {
                    bottom_up += 6;
                } else if ds < 0 {
                    top_down_peel += 6;
                }

                let prev_peel = matches!(
                    prev.op(),
                    CognitiveOpKind::DecomposePeel | CognitiveOpKind::RouteSubHead
                );
                let curr_fuse = matches!(
                    curr.op(),
                    CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit
                );
                if prev_peel && curr_fuse {
                    bottom_up += 10;
                }
                if matches!(curr.op(), CognitiveOpKind::DecomposePeel)
                    && matches!(prev.op(), CognitiveOpKind::DecomposePeel)
                {
                    top_down_peel += 8;
                }
                if matches!(curr.op(), CognitiveOpKind::RepulseAudit) {
                    repulse_streak += if i == 0 { 10 } else { 6 };
                }
            }
            i += 1;
        }

        let bu = if bottom_up > 48 { 48 } else { bottom_up as u8 };
        let td = if top_down_peel > 48 {
            48
        } else {
            top_down_peel as u8
        };
        let rs = if repulse_streak > 48 {
            48
        } else {
            repulse_streak as u8
        };
        let vel = if scale_vel > 64 {
            64
        } else if scale_vel < -64 {
            -64
        } else {
            scale_vel as i8
        };
        (bu, td, rs, vel)
    }

    /// Evaluates the stigmergic gating activation of a cognitive circuit at `scale` using branchless
    /// integer SWAR phase closeness between `input_words` and `circuit_bias_words`, modulated by
    /// the trajectory ledger's recent transitions and temporal synaptic momentum.
    #[inline(always)]
    #[must_use]
    pub fn evaluate_circuit_activation(
        &self,
        circuit_bias_words: &[u64; NUM_STIGMERGIC_GRANULES],
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        scale: StigmergicScaleGranularity,
    ) -> CognitiveCircuitActivation {
        let cell_idx = (scale as usize) & (NUM_STIGMERGIC_GRANULES - 1);
        let diff_local =
            swar_q7_circular_diff_x8(input_words[cell_idx], circuit_bias_words[cell_idx]);
        let abs_local = swar_q7_abs_u8_x8(diff_local);
        let mean_diff_local = swar_mean_u8_le128_x8(abs_local);

        let mut total_diff = 0u16;
        let mut c = 0usize;
        while c < NUM_STIGMERGIC_GRANULES {
            let d = swar_q7_circular_diff_x8(input_words[c], circuit_bias_words[c]);
            total_diff += u16::from(swar_mean_u8_le128_x8(swar_q7_abs_u8_x8(d)));
            c += 1;
        }
        let mean_diff_global = ((total_diff + 4) >> 3) as u8;
        // Branchless closeness in `0..=128`: `128` = identical phase, `64` = orthogonal, `0` = antipodal.
        let local_close = 128u8.saturating_sub(mean_diff_local);
        let global_close = 128u8.saturating_sub(mean_diff_global);
        let signed_resonance = (i16::from(local_close) - 64).clamp(-127, 127) as gcso_q7_t;

        let (op_hist, depth) = self.op_counts();
        let scale_hist = self.scale_counts();
        let (bottom_up_mom, top_down_peel_mom, repulse_streak, net_scale_vel) =
            self.synaptic_transition_stats();
        let peel_hist = op_hist[CognitiveOpKind::DecomposePeel as usize]
            .saturating_add(op_hist[CognitiveOpKind::RouteSubHead as usize]);
        let fuse_hist = op_hist[CognitiveOpKind::SynthesizeFuse as usize]
            .saturating_add(op_hist[CognitiveOpKind::CrystallizeCommit as usize]);
        let repulse_hist = op_hist[CognitiveOpKind::RepulseAudit as usize];

        let divergence = local_close.abs_diff(global_close);
        let peel_raw = 44u16
            + u16::from(divergence)
            + u16::from(peel_hist) * 7
            + u16::from(top_down_peel_mom)
            + if scale == StigmergicScaleGranularity::CellTrail {
                10
            } else {
                0
            };
        let fuse_raw = 36u16
            + (u16::from(global_close) >> 1)
            + u16::from(fuse_hist) * 7
            + u16::from(bottom_up_mom)
            + if scale == StigmergicScaleGranularity::MacroCoalition {
                10
            } else {
                0
            };
        let repulse_raw = 28u16
            + (u16::from(128u8.saturating_sub(global_close)) >> 1)
            + u16::from(repulse_hist) * 10
            + u16::from(repulse_streak);

        let peel_gate_q7 = peel_raw.min(128) as u8;
        let fuse_gate_q7 = fuse_raw.min(128) as u8;
        let repulse_sensitivity_q7 = repulse_raw.min(128) as u8;

        // Branchless multi-gate arbitration across neural propensities.
        let repulse_dominant = repulse_sensitivity_q7 >= 96 && global_close <= 36;
        let peel_dominant = peel_gate_q7 > fuse_gate_q7 && peel_gate_q7 >= 56;
        let selected_op = if repulse_dominant {
            CognitiveOpKind::RepulseAudit
        } else if peel_dominant {
            if scale == StigmergicScaleGranularity::CellTrail {
                CognitiveOpKind::DecomposePeel
            } else {
                CognitiveOpKind::RouteSubHead
            }
        } else if divergence >= 28 {
            CognitiveOpKind::BridgeShift
        } else if global_close >= 88 {
            CognitiveOpKind::SynthesizeFuse
        } else {
            CognitiveOpKind::SteerAlign
        };

        // Perspective movement tendency derived from both the selected operation and scale velocity.
        let top_down_bias = scale_hist[StigmergicScaleGranularity::MacroCoalition as usize]
            >= scale_hist[StigmergicScaleGranularity::PointerBit as usize]
            || net_scale_vel > 0;
        let next_scale = match selected_op {
            CognitiveOpKind::DecomposePeel => StigmergicScaleGranularity::PointerBit,
            CognitiveOpKind::RouteSubHead | CognitiveOpKind::BridgeShift => {
                StigmergicScaleGranularity::CellTrail
            }
            CognitiveOpKind::RepulseAudit => StigmergicScaleGranularity::TypeGateway,
            CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit => {
                if top_down_bias {
                    StigmergicScaleGranularity::MacroCoalition
                } else {
                    StigmergicScaleGranularity::TypeGateway
                }
            }
            _ => scale,
        };
        let peak_gate = peel_gate_q7.max(fuse_gate_q7).max(repulse_sensitivity_q7);

        CognitiveCircuitActivation {
            peel_gate_q7,
            fuse_gate_q7,
            repulse_sensitivity_q7,
            selected_op: selected_op as u8,
            preferred_next_scale: next_scale as u8,
            intensity_q7: peak_gate,
            circuit_resonance_q7: signed_resonance,
            trajectory_depth: depth,
        }
    }

    /// Evaluates cross-trajectory phase-circuit resonance between a query's [`CognitiveTrajectoryLedger`]
    /// (`self`) and a draft's [`CognitiveTrajectoryLedger`] (`draft`) over their 64-head phase fields.
    ///
    /// Detects structural reasoning deficits such as compound decomposition omitted in the draft,
    /// guard repulsion streaks, and scale-horizon mismatch for closed-loop self-reflection.
    #[inline(always)]
    #[must_use]
    pub fn evaluate_cross_trajectory_activation(
        &self,
        draft: &Self,
        query_words: &[u64; NUM_STIGMERGIC_GRANULES],
        draft_words: &[u64; NUM_STIGMERGIC_GRANULES],
    ) -> CognitiveCircuitActivation {
        let mut base = draft.evaluate_circuit_activation(
            query_words,
            draft_words,
            StigmergicScaleGranularity::MacroCoalition,
        );
        let (q_ops, q_depth) = self.op_counts();
        let (d_ops, d_depth) = draft.op_counts();
        let (d_bu, _, d_repulse_streak, _) = draft.synaptic_transition_stats();

        let q_peel = q_ops[CognitiveOpKind::DecomposePeel as usize]
            .saturating_add(q_ops[CognitiveOpKind::RouteSubHead as usize]);
        let d_peel = d_ops[CognitiveOpKind::DecomposePeel as usize]
            .saturating_add(d_ops[CognitiveOpKind::RouteSubHead as usize]);
        let peel_deficit = q_peel.saturating_sub(d_peel);

        let d_repulse = d_ops[CognitiveOpKind::RepulseAudit as usize];
        let depth_gap = q_depth.saturating_sub(d_depth);

        let adj_peel = (u16::from(base.peel_gate_q7)
            + u16::from(peel_deficit) * 18
            + u16::from(depth_gap) * 6)
            .min(128) as u8;
        let adj_repulse = (u16::from(base.repulse_sensitivity_q7)
            + u16::from(d_repulse) * 14
            + (u16::from(d_repulse_streak) >> 1))
            .min(128) as u8;
        let adj_fuse = (u16::from(base.fuse_gate_q7) + (u16::from(d_bu) >> 1))
            .saturating_sub(u16::from(peel_deficit) * 12)
            .min(128) as u8;

        base.peel_gate_q7 = adj_peel;
        base.repulse_sensitivity_q7 = adj_repulse;
        base.fuse_gate_q7 = adj_fuse;

        let selected_op = if adj_repulse >= 92 && base.circuit_resonance_q7 <= 16 {
            CognitiveOpKind::RepulseAudit
        } else if peel_deficit > 0 && adj_peel >= adj_fuse {
            CognitiveOpKind::DecomposePeel
        } else if adj_fuse >= 80 && base.circuit_resonance_q7 >= 24 {
            CognitiveOpKind::CrystallizeCommit
        } else {
            base.op_kind()
        };
        base.selected_op = selected_op as u8;
        base.preferred_next_scale = match selected_op {
            CognitiveOpKind::DecomposePeel => StigmergicScaleGranularity::CellTrail as u8,
            CognitiveOpKind::RepulseAudit => StigmergicScaleGranularity::TypeGateway as u8,
            CognitiveOpKind::CrystallizeCommit | CognitiveOpKind::SynthesizeFuse => {
                StigmergicScaleGranularity::MacroCoalition as u8
            }
            _ => base.preferred_next_scale,
        };
        base.intensity_q7 = adj_peel.max(adj_fuse).max(adj_repulse);
        base
    }

    /// Specializes [`StigmergicThresholds`] directly from this cognitive circuit's accumulated
    /// operation history (`DecomposePeel` vs `SynthesizeFuse` vs `RepulseAudit`) and active `scale`.
    ///
    /// Keeps all regime thresholds strictly positive and monotonic
    /// (`express <= subhead <= rank1 <= crystallize`) while allowing identical runtime inputs
    /// to fire different `ActionHub` regimes according to learned/preset cognitive habits.
    #[inline]
    #[must_use]
    pub fn bias_stigmergic_thresholds(
        &self,
        base: StigmergicThresholds,
        scale: StigmergicScaleGranularity,
    ) -> StigmergicThresholds {
        let (op_hist, depth) = self.op_counts();
        if depth == 0 {
            return base;
        }
        let peel_hist = op_hist[CognitiveOpKind::DecomposePeel as usize]
            .saturating_add(op_hist[CognitiveOpKind::RouteSubHead as usize]);
        let fuse_hist = op_hist[CognitiveOpKind::SynthesizeFuse as usize]
            .saturating_add(op_hist[CognitiveOpKind::CrystallizeCommit as usize]);
        let repulse_hist = op_hist[CognitiveOpKind::RepulseAudit as usize];

        let inv_d = 1.0_f32 / f32::from(depth);
        let peel_ratio = f32::from(peel_hist) * inv_d;
        let fuse_ratio = f32::from(fuse_hist) * inv_d;
        let repulse_ratio = f32::from(repulse_hist) * inv_d;

        let scale_macro = match scale {
            StigmergicScaleGranularity::MacroCoalition => 1.0_f32,
            StigmergicScaleGranularity::TypeGateway => 0.75_f32,
            StigmergicScaleGranularity::CellTrail => 0.50_f32,
            StigmergicScaleGranularity::PointerBit => 0.25_f32,
        };
        let scale_micro = 1.25_f32 - scale_macro;

        let gain_mul = 1.0_f32 + 0.15_f32 * fuse_ratio;
        let skip_mul = branchless_clamp_f32(
            1.0_f32 - 0.20_f32 * peel_ratio - 0.20_f32 * repulse_ratio,
            0.40,
            1.25,
        );
        let subhead_mul =
            branchless_clamp_f32(1.0_f32 - 0.22_f32 * peel_ratio * scale_micro, 0.55, 1.15);
        let macro_mul =
            branchless_clamp_f32(1.0_f32 - 0.22_f32 * fuse_ratio * scale_macro, 0.55, 1.15);
        let crystal_peel_guard = 1.0_f32 + 0.15_f32 * peel_ratio;
        let contradict_mul = branchless_clamp_f32(1.0_f32 - 0.30_f32 * repulse_ratio, 0.45, 1.10);

        let express = branchless_clamp_f32(base.express_steer_density_thresh, 0.05, 2.0);
        let subhead = branchless_clamp_f32(
            base.subhead_route_density_thresh * subhead_mul,
            express + 0.05,
            3.5,
        );
        let rank1 = branchless_clamp_f32(
            base.rank1_inject_density_thresh * macro_mul,
            subhead + 0.05,
            4.5,
        );
        let crystallize = branchless_clamp_f32(
            base.crystallize_density_thresh * macro_mul * crystal_peel_guard,
            rank1 + 0.05,
            6.5,
        );

        StigmergicThresholds {
            evaporation_retain: base.evaporation_retain,
            reinforcement_gain: branchless_clamp_f32(
                base.reinforcement_gain * gain_mul,
                0.05,
                1.50,
            ),
            inertial_skip_entropy_thresh: branchless_clamp_f32(
                base.inertial_skip_entropy_thresh * skip_mul,
                0.03,
                0.50,
            ),
            express_steer_density_thresh: express,
            subhead_route_density_thresh: subhead,
            rank1_inject_density_thresh: rank1,
            crystallize_density_thresh: crystallize,
            contradiction_coherence_thresh: branchless_clamp_f32(
                base.contradiction_coherence_thresh * contradict_mul,
                0.05,
                0.85,
            ),
        }
    }

    /// Packs this [`CognitiveTrajectoryLedger`] into a 64-byte C-ABI [`gcso_ledger_slot_t`]
    /// marked with [`LEDGER_FLAG_COGNITIVE_TRAJECTORY`] for `.gcso` persistence.
    #[inline(always)]
    #[must_use]
    pub fn to_persona_ledger_slot(&self) -> gcso_ledger_slot_t {
        let mut slot = gcso_ledger_slot_t::default();
        self.write_to_ledger_slot(&mut slot);
        slot.telemetry_ring_head = u16::from(self.op_counts().1);
        slot
    }
}

/// Trait governing domain-agnostic long-term plasticity and compositional blending over
/// register-resident stigmergic cognitive circuits.
///
/// Updates are driven strictly by geometric convergence and circuit activation invariants so
/// learned reasoning habits remain invariant across datasets and domains.
pub trait CognitiveCircuitPlasticity: Sized {
    /// Updates the circuit from a completed [`CognitiveCircuitActivation`] and convergence flag
    /// using bounded homeostatic plasticity (LTP on coherent convergence, LTD/shift on drift).
    fn reinforce_from_activation(
        &mut self,
        activation: &CognitiveCircuitActivation,
        converged: bool,
    );

    /// Blends `self` (base persona circuit) with `overlay` (adapted session circuit) according to
    /// `blend_q7` (`0..=128`), preserving active scale transitions from both circuits.
    fn compose_persona_circuit(&self, overlay: &Self, blend_q7: u8) -> Self;
}

impl CognitiveCircuitPlasticity for CognitiveTrajectoryLedger {
    #[inline]
    fn reinforce_from_activation(
        &mut self,
        activation: &CognitiveCircuitActivation,
        converged: bool,
    ) {
        if converged && activation.circuit_resonance_q7 >= 16 {
            let op = activation.op_kind();
            let scale = activation.next_scale();
            let target_intensity = (3 + ((activation.intensity_q7 >> 5) & 3)).min(7);
            if let Some(latest) = self.latest_step() {
                if latest.op() == op && latest.granularity() == scale {
                    // Homeostatic saturation: strengthen in-place rather than flooding all 8 slots.
                    let boosted = latest.intensity().saturating_add(1).min(7);
                    self.steps[0] = CognitiveStepOp::pack(scale, op, boosted).0;
                    return;
                }
            }
            self.push_step(CognitiveStepOp::pack(scale, op, target_intensity));
        } else if !converged {
            let corrective_op = if activation.repulse_sensitivity_q7 >= 84 {
                CognitiveOpKind::RepulseAudit
            } else {
                CognitiveOpKind::BridgeShift
            };
            let intensity = (4 + ((activation.intensity_q7 >> 6) & 3)).min(7);
            self.push_step(CognitiveStepOp::pack(
                activation.next_scale(),
                corrective_op,
                intensity,
            ));
        }
    }

    #[inline]
    fn compose_persona_circuit(&self, overlay: &Self, blend_q7: u8) -> Self {
        let w_overlay = u16::from(blend_q7.min(128));
        if w_overlay == 0 {
            return *self;
        }
        if w_overlay == 128 {
            return *overlay;
        }
        let w_base = 128u16 - w_overlay;
        let mut out_steps = [0u8; LEDGER_HISTORY_LEN];
        let mut i = 0usize;
        while i < LEDGER_HISTORY_LEN {
            let b_step = CognitiveStepOp(self.steps[i]);
            let o_step = CognitiveStepOp(overlay.steps[i]);
            out_steps[i] = match (b_step.is_active(), o_step.is_active()) {
                (false, false) => 0,
                (true, false) => {
                    let scaled = ((u16::from(b_step.intensity()) * w_base) >> 7).clamp(1, 7) as u8;
                    CognitiveStepOp::pack(b_step.granularity(), b_step.op(), scaled).0
                }
                (false, true) => {
                    let scaled =
                        ((u16::from(o_step.intensity()) * w_overlay) >> 7).clamp(1, 7) as u8;
                    CognitiveStepOp::pack(o_step.granularity(), o_step.op(), scaled).0
                }
                (true, true) => {
                    let b_score = u16::from(b_step.intensity()) * w_base;
                    let o_score = u16::from(o_step.intensity()) * w_overlay;
                    let blended_int = (((b_score + o_score) >> 7) as u8).clamp(1, 7);
                    let dominant = if o_score > b_score { o_step } else { b_step };
                    CognitiveStepOp::pack(dominant.granularity(), dominant.op(), blended_int).0
                }
            };
            i += 1;
        }
        Self { steps: out_steps }
    }
}

impl RawPhaseProjectable for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        words_to_q7_array_64(&self.to_phase_words_64())
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_phase_words_64()
    }
}

/// Trait abstracting scalable, host-RAM-resident stigmergic cognitive circuit banks built out of
/// uniform 8-byte [`CognitiveTrajectoryLedger`] units.
///
/// Allows CPU/RAM capacity to scale the effective reasoning horizon (e.g. `N = 8, 32, 128` units
/// for 64 to 1,024+ cognitive steps) with zero disk I/O and zero GPU VRAM overhead, folding the
/// active ensemble down to a single 8-byte activation or ledger at the Hot-Path boundary.
pub trait CognitiveCircuitBankOps {
    /// Returns an immutable slice over all [`CognitiveTrajectoryLedger`] units in this bank.
    fn units(&self) -> &[CognitiveTrajectoryLedger];

    /// Returns a mutable slice over all [`CognitiveTrajectoryLedger`] units in this bank.
    fn units_mut(&mut self) -> &mut [CognitiveTrajectoryLedger];

    /// Pushes `step` into the front working unit (`units[0]`), spilling evicted steps hierarchically
    /// into deeper RAM units without discarding long-horizon context.
    #[inline]
    fn push_step_hierarchical(&mut self, step: CognitiveStepOp) {
        let units = self.units_mut();
        if units.is_empty() || !step.is_active() {
            return;
        }
        let mut carry = step;
        let mut idx = 0usize;
        while idx < units.len() && carry.is_active() {
            let evicted = CognitiveStepOp(units[idx].steps[LEDGER_HISTORY_LEN - 1]);
            units[idx].push_step(carry);
            carry = evicted;
            idx += 1;
        }
    }

    /// Applies homeostatic plasticity across the working unit and consolidates coherent habits
    /// into the next bank tier when `converged` is true.
    #[inline]
    fn reinforce_bank(&mut self, activation: &CognitiveCircuitActivation, converged: bool) {
        let units = self.units_mut();
        if units.is_empty() {
            return;
        }
        units[0].reinforce_from_activation(activation, converged);
        if converged && units.len() > 1 && activation.circuit_resonance_q7 >= 24 {
            if let Some(latest) = units[0].latest_step() {
                units[1].reinforce_from_activation(
                    &CognitiveCircuitActivation {
                        selected_op: latest.op() as u8,
                        preferred_next_scale: latest.granularity() as u8,
                        ..*activation
                    },
                    true,
                );
            }
        }
    }

    /// Evaluates all active units on CPU using branchless integer SWAR resonance and folds the
    /// multi-unit ensemble into a single 8-byte [`CognitiveCircuitActivation`].
    #[inline]
    fn evaluate_bank_activation(
        &self,
        bias_words: &[u64; NUM_STIGMERGIC_GRANULES],
        input_words: &[u64; NUM_STIGMERGIC_GRANULES],
        scale: StigmergicScaleGranularity,
    ) -> CognitiveCircuitActivation {
        let units = self.units();
        if units.is_empty() {
            return CognitiveCircuitActivation::default();
        }
        let mut peel_acc = 0u64;
        let mut fuse_acc = 0u64;
        let mut repulse_acc = 0u64;
        let mut res_acc = 0i64;
        let mut weight_sum = 0u64;
        let mut best_weight = 0u64;
        let mut best_act = units[0].evaluate_circuit_activation(bias_words, input_words, scale);
        let mut total_depth = 0u16;

        let mut i = 0usize;
        while i < units.len() {
            let (_, depth) = units[i].op_counts();
            if depth == 0 {
                i += 1;
                continue;
            }
            total_depth = total_depth.saturating_add(u16::from(depth));
            let act = units[i].evaluate_circuit_activation(bias_words, input_words, scale);
            let recency_bonus = if i == 0 { 32u64 } else { 0u64 };
            let res_w = (128i32 + i32::from(act.circuit_resonance_q7)).max(1) as u64;
            let w = res_w + recency_bonus;

            peel_acc += u64::from(act.peel_gate_q7) * w;
            fuse_acc += u64::from(act.fuse_gate_q7) * w;
            repulse_acc += u64::from(act.repulse_sensitivity_q7) * w;
            res_acc += i64::from(act.circuit_resonance_q7) * (w as i64);
            weight_sum += w;
            if w > best_weight {
                best_weight = w;
                best_act = act;
            }
            i += 1;
        }

        if weight_sum == 0 {
            return best_act;
        }
        let peel_gate_q7 = (peel_acc / weight_sum).min(128) as u8;
        let fuse_gate_q7 = (fuse_acc / weight_sum).min(128) as u8;
        let repulse_sensitivity_q7 = (repulse_acc / weight_sum).min(128) as u8;
        let circuit_resonance_q7 = (res_acc / (weight_sum as i64)).clamp(-127, 127) as gcso_q7_t;
        let intensity_q7 = peel_gate_q7.max(fuse_gate_q7).max(repulse_sensitivity_q7);
        CognitiveCircuitActivation {
            peel_gate_q7,
            fuse_gate_q7,
            repulse_sensitivity_q7,
            selected_op: best_act.selected_op,
            preferred_next_scale: best_act.preferred_next_scale,
            intensity_q7,
            circuit_resonance_q7,
            trajectory_depth: total_depth.min(255) as u8,
        }
    }

    /// Folds the multi-unit RAM bank into a single 8-byte [`CognitiveTrajectoryLedger`] suitable
    /// for `.gcso` persistence or `ManifoldGatewaySignal` synthesis.
    #[inline]
    fn to_consolidated_ledger(&self) -> CognitiveTrajectoryLedger {
        let units = self.units();
        if units.is_empty() {
            return CognitiveTrajectoryLedger::new();
        }
        let mut acc = units[0];
        let mut active_count = 1u16;
        let mut i = 1usize;
        while i < units.len() {
            if units[i].op_counts().1 > 0 {
                let blend = (128u16 / (active_count + 1)).clamp(1, 64) as u8;
                acc = acc.compose_persona_circuit(&units[i], blend);
                active_count += 1;
            }
            i += 1;
        }
        acc
    }
}

impl CognitiveCircuitBankOps for [CognitiveTrajectoryLedger] {
    #[inline(always)]
    fn units(&self) -> &[CognitiveTrajectoryLedger] {
        self
    }

    #[inline(always)]
    fn units_mut(&mut self) -> &mut [CognitiveTrajectoryLedger] {
        self
    }
}

/// Fixed-capacity, cacheline-aligned RAM-resident bank of `N` 8-byte [`CognitiveTrajectoryLedger`]
/// units (`8 * N` bytes total; zero heap allocation, zero VRAM footprint).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CognitiveCircuitBank<const N: usize> {
    /// Contiguous array of 8-byte stigmergic cognitive circuit units (`index 0` = newest working memory).
    pub units: [CognitiveTrajectoryLedger; N],
}

impl<const N: usize> Default for CognitiveCircuitBank<N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> CognitiveCircuitBank<N> {
    /// Creates an empty [`CognitiveCircuitBank`] of `N` units (`8 * N` steps capacity).
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            units: [CognitiveTrajectoryLedger::new(); N],
        }
    }

    /// Seeds the working unit (`units[0]`) from a persisted 8-byte [`CognitiveTrajectoryLedger`].
    #[inline(always)]
    #[must_use]
    pub const fn from_ledger(ledger: CognitiveTrajectoryLedger) -> Self {
        let mut bank = Self::new();
        if N > 0 {
            bank.units[0] = ledger;
        }
        bank
    }
}

impl<const N: usize> CognitiveCircuitBankOps for CognitiveCircuitBank<N> {
    #[inline(always)]
    fn units(&self) -> &[CognitiveTrajectoryLedger] {
        &self.units
    }

    #[inline(always)]
    fn units_mut(&mut self) -> &mut [CognitiveTrajectoryLedger] {
        &mut self.units
    }
}

impl<const N: usize> RawPhaseProjectable for CognitiveCircuitBank<N> {
    #[inline]
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.to_consolidated_ledger().project_to_q7_64()
    }

    #[inline]
    fn project_to_words_64(&self) -> [u64; NUM_STIGMERGIC_GRANULES] {
        self.to_consolidated_ledger().to_phase_words_64()
    }
}

/// Low-spec PC preset: 8 units = 64 bytes (single L1D cacheline, 64 cognitive steps).
pub type LowSpecCognitiveBank = CognitiveCircuitBank<8>;

/// Standard consumer PC preset: 32 units = 256 bytes (256 cognitive steps).
pub type StandardCognitiveBank = CognitiveCircuitBank<32>;

/// High-capacity RAM / Zen 4 L2 preset: 128 units = 1,024 bytes (1,024 cognitive steps).
pub type HighCapCognitiveBank = CognitiveCircuitBank<128>;
