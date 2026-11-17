// name: src/core/src/traits/gateway/circuit.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Multi-scale swarm hierarchy, [`CognitiveTrajectoryLedger`], [`HybridScaleCircuit`],
//! [`CognitiveCircuitBankOps`], and [`StigmergicScaleLayer`] abstractions.

use super::signals::*;
use super::*;

// ===================================================================
// 11. Multi-Scale Swarm Hierarchy, Stigmergic Layer Management &
//     Stigmergic Cognitive Phase Circuit
// ===================================================================

/// Multi-scale granularity tier within the self-similar GCSO hierarchy.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum SwarmScaleGranularity {
    /// Scale 0: 64-bit tagged pointer and single 8-head `u64` SWAR register.
    PointerBit = 0,
    /// Scale 1: 128-byte pointer trail (`8` granules x `8` heads) with local inhibition.
    CellTrail = 1,
    /// Scale 2: Type-level blanket composition and 128-byte [`GatewayFrame`] interference.
    #[default]
    TypeGateway = 2,
    /// Scale 3..N: Macro-level multi-hop pointer ring and cross-stratum coalition cascade.
    MacroCoalition = 3,
}

/// Primary `S`-initial alias for [`SwarmScaleGranularity`] emphasizing stigmergic multi-scale
/// fragmentation (`0..=3`).
pub type StigmergicScaleGranularity = SwarmScaleGranularity;

impl SwarmScaleGranularity {
    /// Converts a raw 2-bit integer (`0..=3`) into a [`SwarmScaleGranularity`].
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
                if v > 7 { 7 } else { v }
            }
        };
        if adj < 2 { 2 } else { adj }
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
        if scaled > 40 { 40 } else { scaled }
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
        if out < 24 { 24 } else { out }
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
        granularity: SwarmScaleGranularity,
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

    /// Unpacks the 2-bit [`SwarmScaleGranularity`].
    #[inline(always)]
    #[must_use]
    pub const fn granularity(self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::from_u8(self.0 >> 6)
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
        let mag = (16i16) + ((self.intensity() as i16) * 8) + (((self.granularity() as u8) as i16) * 12);
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
    /// Preferred next [`SwarmScaleGranularity`] discriminant (`0..=3`) for perspective transition.
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

    /// Returns the typed preferred next [`SwarmScaleGranularity`].
    #[inline(always)]
    #[must_use]
    pub const fn next_scale(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::from_u8(self.preferred_next_scale)
    }
}

/// 8-byte (`#[repr(C, align(8))]`) chronological trajectory ledger encoding the last 8
/// multi-scale cognitive operations (`[Scale | Op | Intensity]`) inside a single `u64` register.
///
/// Mirrors [`gcso_ledger_slot_t::telemetry_mini_ledger`] 1:1 so observed scale transitions and
/// reasoning habits act directly as reusable persona circuit parameters via
/// [`ManifoldGatewaySignal`] and [`StigmergicScaleLayer`].
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
pub(super) const fn swar_mean_u8_le128_x8(w: u64) -> u8 {
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

    /// Returns the count of recorded steps at each [`SwarmScaleGranularity`] level (`0..4`).
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
    pub const fn to_phase_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        let mut words = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut c = 0usize;
        while c < NUM_SWARM_MICRO_CELLS {
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
        circuit_bias_words: &[u64; NUM_SWARM_MICRO_CELLS],
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
        scale: SwarmScaleGranularity,
    ) -> CognitiveCircuitActivation {
        let cell_idx = (scale as usize) & (NUM_SWARM_MICRO_CELLS - 1);
        let diff_local = swar_q7_circular_diff_x8(input_words[cell_idx], circuit_bias_words[cell_idx]);
        let abs_local = swar_q7_abs_u8_x8(diff_local);
        let mean_diff_local = swar_mean_u8_le128_x8(abs_local);

        let mut total_diff = 0u16;
        let mut c = 0usize;
        while c < NUM_SWARM_MICRO_CELLS {
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
            + if scale == SwarmScaleGranularity::CellTrail { 10 } else { 0 };
        let fuse_raw = 36u16
            + (u16::from(global_close) >> 1)
            + u16::from(fuse_hist) * 7
            + u16::from(bottom_up_mom)
            + if scale == SwarmScaleGranularity::MacroCoalition { 10 } else { 0 };
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
            if scale == SwarmScaleGranularity::CellTrail {
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
        let top_down_bias = scale_hist[SwarmScaleGranularity::MacroCoalition as usize]
            >= scale_hist[SwarmScaleGranularity::PointerBit as usize]
            || net_scale_vel > 0;
        let next_scale = match selected_op {
            CognitiveOpKind::DecomposePeel => SwarmScaleGranularity::PointerBit,
            CognitiveOpKind::RouteSubHead | CognitiveOpKind::BridgeShift => {
                SwarmScaleGranularity::CellTrail
            }
            CognitiveOpKind::RepulseAudit => SwarmScaleGranularity::TypeGateway,
            CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit => {
                if top_down_bias {
                    SwarmScaleGranularity::MacroCoalition
                } else {
                    SwarmScaleGranularity::TypeGateway
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
        query_words: &[u64; NUM_SWARM_MICRO_CELLS],
        draft_words: &[u64; NUM_SWARM_MICRO_CELLS],
    ) -> CognitiveCircuitActivation {
        let mut base = draft.evaluate_circuit_activation(
            query_words,
            draft_words,
            SwarmScaleGranularity::MacroCoalition,
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
            CognitiveOpKind::DecomposePeel => SwarmScaleGranularity::CellTrail as u8,
            CognitiveOpKind::RepulseAudit => SwarmScaleGranularity::TypeGateway as u8,
            CognitiveOpKind::CrystallizeCommit | CognitiveOpKind::SynthesizeFuse => {
                SwarmScaleGranularity::MacroCoalition as u8
            }
            _ => base.preferred_next_scale,
        };
        base.intensity_q7 = adj_peel.max(adj_fuse).max(adj_repulse);
        base
    }

    /// Evaluates this trajectory circuit against `(frm_w, trail_w)` at `scale` and modulates
    /// `frame`'s PSPM head allocation, sub-head gains, EDBC inverse-temperature scale, and
    /// anti-resonance guard severity in place without dataset-specific constants.
    ///
    /// When the trajectory ledger is empty (`trajectory_depth == 0`), `frame` is returned unmodified.
    #[inline]
    pub fn modulate_gateway_frame_at_scale(
        &self,
        frame: &mut GatewayFrame,
        frm_w: &[u64; NUM_SWARM_MICRO_CELLS],
        trail_w: &[u64; NUM_SWARM_MICRO_CELLS],
        scale: SwarmScaleGranularity,
    ) -> CognitiveCircuitActivation {
        let act = self.evaluate_circuit_activation(frm_w, trail_w, scale);
        if act.trajectory_depth == 0 {
            return act;
        }
        const MIN_BUCKET_HEADS: u8 = 8;
        match act.op_kind() {
            CognitiveOpKind::DecomposePeel | CognitiveOpKind::RouteSubHead => {
                let shift = ((act.peel_gate_q7.saturating_sub(56)) >> 3).min(8);
                let take_logic = (shift >> 1)
                    .min(frame.subhead_logic_heads.saturating_sub(MIN_BUCKET_HEADS));
                let rem = shift.saturating_sub(take_logic);
                let take_explore =
                    rem.min(frame.subhead_explore_heads.saturating_sub(MIN_BUCKET_HEADS));
                let gained = take_logic.saturating_add(take_explore);
                frame.subhead_logic_heads = frame.subhead_logic_heads.saturating_sub(take_logic);
                frame.subhead_explore_heads =
                    frame.subhead_explore_heads.saturating_sub(take_explore);
                frame.subhead_fact_heads = frame.subhead_fact_heads.saturating_add(gained);
                frame.subhead_fact_gain_q7 =
                    frame.subhead_fact_gain_q7.saturating_add(act.peel_gate_q7 >> 3);
            }
            CognitiveOpKind::SynthesizeFuse | CognitiveOpKind::CrystallizeCommit => {
                let shift = ((act.fuse_gate_q7.saturating_sub(56)) >> 3).min(8);
                let take_fact = (shift >> 1)
                    .min(frame.subhead_fact_heads.saturating_sub(MIN_BUCKET_HEADS));
                let rem = shift.saturating_sub(take_fact);
                let take_explore =
                    rem.min(frame.subhead_explore_heads.saturating_sub(MIN_BUCKET_HEADS));
                let gained = take_fact.saturating_add(take_explore);
                frame.subhead_fact_heads = frame.subhead_fact_heads.saturating_sub(take_fact);
                frame.subhead_explore_heads =
                    frame.subhead_explore_heads.saturating_sub(take_explore);
                frame.subhead_logic_heads = frame.subhead_logic_heads.saturating_add(gained);
                frame.subhead_logic_gain_q7 =
                    frame.subhead_logic_gain_q7.saturating_add(act.fuse_gate_q7 >> 3);
                let boost = 1.0_f32 + (f32::from(act.fuse_gate_q7) * (0.25_f32 / 128.0_f32));
                frame.effective_inv_temp =
                    branchless_clamp_f32(frame.effective_inv_temp * boost, 0.25, 4.0);
            }
            CognitiveOpKind::RepulseAudit => {
                let corroborated = act.circuit_resonance_q7 < 0
                    || frame.fact_persona_coherence < 0.0
                    || frame.repulsion_severity > 0.25;
                if corroborated {
                    let sev_floor = branchless_clamp_f32(
                        f32::from(act.repulse_sensitivity_q7) / 128.0_f32,
                        0.55,
                        1.0,
                    );
                    frame.repulsion_severity =
                        branchless_max_f32(frame.repulsion_severity, sev_floor);
                    frame.action_mode_flags |= ACTION_MODE_PHASE_REPULSE;
                    frame.provenance_bits |= CAUSAL_CAUSE_COHOMOLOGY_REPULSION;
                    frame.subhead_explore_gain_q7 = frame
                        .subhead_explore_gain_q7
                        .saturating_add(act.repulse_sensitivity_q7 >> 3);
                }
            }
            CognitiveOpKind::BridgeShift | CognitiveOpKind::SteerAlign | CognitiveOpKind::Observe => {
                let inc = act.intensity_q7 >> 4;
                frame.subhead_fact_gain_q7 = frame.subhead_fact_gain_q7.saturating_add(inc);
                frame.subhead_logic_gain_q7 = frame.subhead_logic_gain_q7.saturating_add(inc);
            }
        }
        act
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
        scale: SwarmScaleGranularity,
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
            SwarmScaleGranularity::MacroCoalition => 1.0_f32,
            SwarmScaleGranularity::TypeGateway => 0.75_f32,
            SwarmScaleGranularity::CellTrail => 0.50_f32,
            SwarmScaleGranularity::PointerBit => 0.25_f32,
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
                    let scaled =
                        ((u16::from(b_step.intensity()) * w_base) >> 7).clamp(1, 7) as u8;
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
                    let blended_int =
                        (((b_score + o_score) >> 7) as u8).clamp(1, 7);
                    let dominant = if o_score > b_score { o_step } else { b_step };
                    CognitiveStepOp::pack(dominant.granularity(), dominant.op(), blended_int).0
                }
            };
            i += 1;
        }
        Self { steps: out_steps }
    }
}

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
    pub const fn circuit_for_scale(&self, granularity: SwarmScaleGranularity) -> &CognitiveTrajectoryLedger {
        match granularity {
            SwarmScaleGranularity::PointerBit | SwarmScaleGranularity::CellTrail => {
                &self.micro_circuit
            }
            SwarmScaleGranularity::TypeGateway | SwarmScaleGranularity::MacroCoalition => {
                &self.macro_circuit
            }
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
    fn project_to_q7_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.to_blended_ledger(64).project_to_q7_64()
    }

    #[inline(always)]
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
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
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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
        bias_words: &[u64; NUM_SWARM_MICRO_CELLS],
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
        scale: SwarmScaleGranularity,
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
    /// for `.gcso` persistence or [`ManifoldGatewaySignal`] synthesis.
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
    fn project_to_words_64(&self) -> [u64; NUM_SWARM_MICRO_CELLS] {
        self.to_consolidated_ledger().to_phase_words_64()
    }
}

impl<const N: usize> CausalProvenanceSignal for CognitiveCircuitBank<N> {
    #[inline]
    fn causal_provenance_bits(&self) -> u32 {
        self.to_consolidated_ledger().causal_provenance_bits()
    }
}

impl<const N: usize> CognitiveGroundingSignal for CognitiveCircuitBank<N> {
    #[inline]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.project_to_q7_64()
    }

    #[inline]
    fn grounding_base_energy(&self) -> f32 {
        self.to_consolidated_ledger().grounding_base_energy()
    }
}

impl<const N: usize> SubjectivePersonaSignal for CognitiveCircuitBank<N> {}

impl<const N: usize> ManifoldGatewaySignal for CognitiveCircuitBank<N> {
    #[inline]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_consolidated_ledger().synthesize_gateway_frame()
    }
}

/// Low-spec PC preset: 8 units = 64 bytes (single L1D cacheline, 64 cognitive steps).
pub type LowSpecCognitiveBank = CognitiveCircuitBank<8>;

/// Standard consumer PC preset: 32 units = 256 bytes (256 cognitive steps).
pub type StandardCognitiveBank = CognitiveCircuitBank<32>;

/// High-capacity RAM / Zen 4 L2 preset: 128 units = 1,024 bytes (1,024 cognitive steps).
pub type HighCapCognitiveBank = CognitiveCircuitBank<128>;


/// Output of projecting an input SWAR phase field through one or more [`StigmergicScaleLayer`] tiers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleLayerProjection {
    /// Composite 128-byte [`GatewayFrame`] synthesized at this layer boundary.
    pub frame: GatewayFrame,
    /// Orthogonal contribution explained/peeled by this layer across 64 heads.
    pub peeled_words: [u64; NUM_SWARM_MICRO_CELLS],
    /// Remaining unexplained Q7 phase residual forwarded to layer `layer_index + 1`.
    pub residual_words: [u64; NUM_SWARM_MICRO_CELLS],
    /// Granularity tier executed at this projection step.
    pub granularity: SwarmScaleGranularity,
    /// Zero-based starting layer index across the active stack (`0..total_layers`).
    pub layer_index: usize,
    /// Total effective layer count $N$ across the active cascade.
    pub total_layers: usize,
    /// Stigmergic cognitive gating activation computed from trajectory history and phase resonance.
    pub activation: CognitiveCircuitActivation,
    /// Compact 1-byte cognitive step emitted for recording into [`CognitiveTrajectoryLedger`].
    pub emitted_step: CognitiveStepOp,
}

/// Branchless core helper projecting `signal` at a specific [`SwarmScaleGranularity`] and layer
/// depth ratio `layer_index / total_layers` with zero dynamic allocations.
#[inline]
pub fn project_gateway_signal_at_scale<S: ManifoldGatewaySignal + ?Sized>(
    signal: &S,
    granularity: SwarmScaleGranularity,
    input_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
        SwarmScaleGranularity::PointerBit => 1u8 << (layer_index & 7),
        SwarmScaleGranularity::CellTrail => {
            if frame.active_swarm_cell_mask == 0 {
                0x0F
            } else {
                frame.active_swarm_cell_mask
            }
        }
        SwarmScaleGranularity::TypeGateway | SwarmScaleGranularity::MacroCoalition => 0xFF,
    };

    let mut peeled_words = [0u64; NUM_SWARM_MICRO_CELLS];
    let mut residual_words = *input_words;
    let mut c = 0usize;
    while c < NUM_SWARM_MICRO_CELLS {
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
    /// Returns the primary [`SwarmScaleGranularity`] owned by this layer.
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::TypeGateway
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
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
    /// Assigned [`SwarmScaleGranularity`] tier inside a multi-layer stack.
    pub granularity: SwarmScaleGranularity,
}

impl<T> ScaleLayerNode<T> {
    /// Wraps `signal` at `granularity`.
    #[inline(always)]
    #[must_use]
    pub const fn new(signal: T, granularity: SwarmScaleGranularity) -> Self {
        Self { signal, granularity }
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
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        self.signal.synthesize_gateway_frame_with_words()
    }
}

impl<T: ManifoldGatewaySignal> StigmergicScaleLayer for ScaleLayerNode<T> {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        self.granularity
    }
}

impl<T: StigmergicScaleLayer + ?Sized> StigmergicScaleLayer for &T {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        (**self).scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        (**self).layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        (**self).project_scale_layer(input_words, ledger, layer_index, total_layers)
    }
}

/// Composite 2-layer [`StigmergicScaleLayer`] adapter chaining any lower layer `A` into
/// upper layer `B` via orthogonal residual peeling with zero heap allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayeredScalePair<A, B>(
    /// First (typically coarser or lower-tier) scale layer.
    pub A,
    /// Second layer receiving the unpeeled orthogonal residual from `self.0`.
    pub B,
);

impl<A: StigmergicScaleLayer, B: StigmergicScaleLayer> CausalProvenanceSignal
    for LayeredScalePair<A, B>
{
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        self.0.causal_provenance_bits() | self.1.causal_provenance_bits()
    }

    #[inline(always)]
    fn causal_anomaly_severity(&self) -> f32 {
        branchless_max_f32(self.0.causal_anomaly_severity(), self.1.causal_anomaly_severity())
    }
}

impl<A: StigmergicScaleLayer, B: StigmergicScaleLayer> ManifoldGatewaySignal
    for LayeredScalePair<A, B>
{
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let (f0, w0) = self.0.synthesize_gateway_frame_with_words();
        let (f1, w1) = self.1.synthesize_gateway_frame_with_words();
        let uni0 = is_word_array_uniform_64(&w0);
        let uni1 = is_word_array_uniform_64(&w1);
        let e0 = branchless_flush_subnormal_f32(branchless_relu_f32(f0.effective_energy));
        let e1 = branchless_flush_subnormal_f32(branchless_relu_f32(f1.effective_energy));
        let tot = e0 + e1;
        let s_u64 = if tot > DEFAULT_ENTROPY_EPS {
            let ratio = branchless_clamp_f32(e1 * fast_inv_pos_normal_f32(tot), 0.0, 1.0);
            ((ratio * 128.0 + 0.5) as u64).min(128)
        } else {
            64
        };
        let bias = (128u64.wrapping_sub(s_u64)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let (fused_w, fused_uni) = swar_slerp_words_with_uniform_flag_64(&w0, uni0, &w1, uni1, s_u64, bias);
        let rep = branchless_max_f32(f0.repulsion_severity, f1.repulsion_severity);
        let tag_q7 = swar_q7_mean_i8_from_words_with_uniform_flag_64(&fused_w, fused_uni);
        let dom = if e1 > e0 { f1 } else { f0 };
        let prov = f0.provenance_bits | f1.provenance_bits;
        (
            GatewayFrame {
                tagged_ptr: dom.tagged_ptr,
                mapped_section_offset48: dom.mapped_section_offset48,
                target_anchor_id: dom.target_anchor_id,
                effective_energy: branchless_min_f32(tot, MAX_ATTRACTOR_PULL_FORCE),
                repulsion_severity: rep,
                provenance_bits: prov,
                dominant_cause_code: f0.dominant_cause_code.max(f1.dominant_cause_code),
                dominant_slot: dom.dominant_slot,
                action_mode_flags: f0.action_mode_flags | f1.action_mode_flags,
                active_swarm_cell_mask: f0.active_swarm_cell_mask | f1.active_swarm_cell_mask,
                mean_phase_tag_q7: tag_q7,
                phase_deltas: words_to_q7_array_64(&fused_w),
                ..dom
            },
            fused_w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl<A: StigmergicScaleLayer, B: StigmergicScaleLayer> StigmergicScaleLayer
    for LayeredScalePair<A, B>
{
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
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
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
        let mut combined_peeled = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut c = 0usize;
        while c < NUM_SWARM_MICRO_CELLS {
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

impl<
        A: StigmergicScaleLayer + ObjectiveFactSignal,
        B: StigmergicScaleLayer + SubjectivePersonaSignal,
    > StigmergicScaleLayer for (A, B)
{
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
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
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
        let mut combined_peeled = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut c = 0usize;
        while c < NUM_SWARM_MICRO_CELLS {
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
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        if self.is_empty() {
            SwarmScaleGranularity::TypeGateway
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
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        if self.is_empty() {
            return ScaleLayerProjection {
                frame: GatewayFrame::default(),
                peeled_words: [0u64; NUM_SWARM_MICRO_CELLS],
                residual_words: *input_words,
                granularity: SwarmScaleGranularity::TypeGateway,
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
        let mut combined_peeled = [0u64; NUM_SWARM_MICRO_CELLS];
        let mut cursor = layer_index;
        let mut last_proj = self[0].project_scale_layer(&cur_residual, &cur_ledger, cursor, total);
        let mut c = 0usize;
        while c < NUM_SWARM_MICRO_CELLS {
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
            while k < NUM_SWARM_MICRO_CELLS {
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
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        self.as_slice().scale_granularity()
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        self.as_slice().layer_count()
    }

    #[inline(always)]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
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
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::TypeGateway
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for ObjectiveFact<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::CellTrail
    }
}

impl<P: RawPhaseProjectable + Clone> StigmergicScaleLayer for SubjectivePersona<P> {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for GuardPhaseAnchor {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::TypeGateway
    }
}

impl StigmergicScaleLayer for CrystallizedAttractorRecord {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::MacroCoalition
    }
}

impl StigmergicScaleLayer for CognitiveTrajectoryLedger {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        self.latest_step()
            .map(CognitiveStepOp::granularity)
            .unwrap_or(SwarmScaleGranularity::TypeGateway)
    }
}

impl StigmergicScaleLayer for HybridScaleCircuit {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        SwarmScaleGranularity::CellTrail
    }

    #[inline(always)]
    fn layer_count(&self) -> usize {
        2
    }

    #[inline]
    fn project_scale_layer(
        &self,
        input_words: &[u64; NUM_SWARM_MICRO_CELLS],
        ledger: &CognitiveTrajectoryLedger,
        layer_index: usize,
        total_layers: usize,
    ) -> ScaleLayerProjection {
        let g = SwarmScaleGranularity::from_layer_ratio(layer_index, total_layers);
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

impl CausalProvenanceSignal for gcso_pointer_trail_t {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        let rep = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let has_anchor = self.target_anchor_id != 0;
        branchless_select_u32(has_anchor, CAUSAL_AXIS_OBJECTIVE_FACT, 0)
            | branchless_select_u32(rep, CAUSAL_AXIS_CONTRADICTION_GUARD | CAUSAL_CAUSE_COHOMOLOGY_REPULSION, 0)
    }
}

impl ManifoldGatewaySignal for gcso_pointer_trail_t {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(&self) -> (GatewayFrame, [u64; NUM_SWARM_MICRO_CELLS]) {
        let w = *trail_phase_words_ref(self);
        let tag_q7 = fast_mean_q7_tag_from_words_64(&w);
        let ptr = gcso_tagged_ptr_t(self.current_ptr);
        let rep = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) != 0;
        let dom_slot = ptr.unpack_tag() & ATTRACTOR_SLOT_ID_MASK;
        let action_flags = branchless_select_u8(
            rep,
            ACTION_PRESET_REPULSION_ESCAPE,
            ACTION_PRESET_EXPRESS_PHASE,
        );
        let packed_tag = pack_action_slot_tag(dom_slot, action_flags);
        (
            GatewayFrame {
                tagged_ptr: self.current_ptr,
                mapped_section_offset48: ptr.unpack_addr(),
                target_anchor_id: self.target_anchor_id,
                effective_energy: branchless_min_f32(
                    branchless_max_f32(self.attractor_pull_force, self.stigmergic_density),
                    MAX_ATTRACTOR_PULL_FORCE,
                ),
                repulsion_severity: branchless_select_f32(rep, 0.85, 0.0),
                provenance_bits: self.causal_provenance_bits(),
                dominant_cause_code: branchless_select_u8(
                    rep,
                    ROOT_CAUSE_COHOMOLOGY_REPULSION,
                    ROOT_CAUSE_NOMINAL,
                ),
                dominant_slot: dom_slot,
                action_mode_flags: action_flags,
                active_swarm_cell_mask: ((self.flags & crate::traits::telemetry::TRAIL_SWARM_QUORUM_MASK)
                    >> crate::traits::telemetry::TRAIL_SWARM_QUORUM_SHIFT) as u8,
                mean_phase_tag_q7: tag_q7,
                active_head_count: MAX_ATTENTION_HEADS as u16,
                frame_flags: packed_tag as u16,
                phase_deltas: self.accumulated_phase_delta,
                ..GatewayFrame::default()
            },
            w,
        )
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.synthesize_gateway_frame_with_words().0
    }
}

impl StigmergicScaleLayer for gcso_pointer_trail_t {
    #[inline(always)]
    fn scale_granularity(&self) -> SwarmScaleGranularity {
        if self.linked_trail_id != 0 {
            SwarmScaleGranularity::MacroCoalition
        } else {
            SwarmScaleGranularity::CellTrail
        }
    }
}

