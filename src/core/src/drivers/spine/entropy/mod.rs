// name: src/core/src/drivers/spine/entropy/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-Driven Decoding Branch Controller (`EntropyController`) and stateless entropy kernels.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use core::f32::consts::LN_2;

use crate::foundation::{Pod, DEFAULT_MAX_VALID_Z_SCORE};
use crate::liminal::DEFAULT_ENTROPY_EPS;

/// `EntropyController` state structure (32 bytes, 32-byte aligned, Canonical Little-Endian `Pod`).
#[repr(C, align(32))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropyState {
    /// Moving Z-score normalized attention entropy ($\tilde{H} \in [0, \ln 2]$, `offset 0..4`).
    pub moving_z_entropy: f32,
    /// Z-score threshold ($\tau_{\mathrm{bifurcation}}$) triggering pitchfork branch split (`offset 4..8`).
    pub bifurcation_threshold: f32,
    /// Numerical safety threshold ($\epsilon_{\mathrm{log}}$) for singularity detection (`offset 8..12`).
    pub singularity_eps: f32,
    /// Sliding-window entropy rate integrator ($\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$, `offset 12..16`).
    pub sliding_entropy_rate: f32,
    /// Gain factor applied to phase-conjugate repulsion pulses ($-\boldsymbol{\Delta\theta}$, `offset 16..20`).
    pub repulsion_gain: f32,
    /// Energy-guided dynamic sampling temperature modifier (`offset 20..24`).
    pub sample_temperature: f32,
    /// Active execution branch mode (`0` = Normal, `1` = Pitchfork Bifurcation, `2` = Repulsion, `offset 24..28`).
    pub active_branch_mode: u32,
    /// Control & status bitmask flags (`offset 28..32`).
    pub flags: u32,
}

impl Default for EntropyState {
    #[inline(always)]
    fn default() -> Self {
        // SAFETY: `EntropyState` is a Canonical Little-Endian `Pod` struct where all-zero bytes are valid.
        unsafe { core::mem::zeroed() }
    }
}

// SAFETY: `EntropyState` is `#[repr(C, align(32))]` containing floats and `u32` fields with zero padding.
unsafe impl Pod for EntropyState {}

const _: () = {
    assert!(core::mem::size_of::<EntropyState>() == 32);
    assert!(core::mem::align_of::<EntropyState>() == 32);
    assert!(core::mem::offset_of!(EntropyState, flags) == 28);
};

/// Normal entropy convergence mode (`Mode 0`: single stable attractor at origin $x^* = 0$).
pub const ENTROPY_GATE_MODE_NORMAL_ENTROPY: u32 = 0;

/// Pitchfork bifurcation exploration mode (`Mode 1`: supercritical double-well $\pm x^*_\pm$).
pub const ENTROPY_GATE_MODE_PITCHFORK_BIFURCATION: u32 = 1;

/// Phase-conjugate repulsion mode (`Mode 2`: anti-phase $-\boldsymbol{\Delta\theta}$ escape).
pub const ENTROPY_GATE_MODE_PHASE_CONJUGATE_REPULSION: u32 = 2;

/// Default logarithmic singularity epsilon fallback constant ($10^{-12}$, `LiminiKa C-ABI`).
pub const DEFAULT_SINGULARITY_EPS: f32 = DEFAULT_ENTROPY_EPS;

/// Default pitchfork bifurcation Z-score threshold constant ($\tau_{\mathrm{bifurcation}} = 1.5$).
pub const DEFAULT_BIFURCATION_THRESHOLD: f32 = 1.5;

/// Default sliding-window entropy rate flux threshold constant ($\tau_{\mathrm{flux}} = 0.5$, Section 3.2).
pub const DEFAULT_TAU_FLUX: f32 = 0.5;

/// Default initial sampling temperature modifier (`1.0`).
pub const DEFAULT_SAMPLE_TEMPERATURE: f32 = 1.0;

/// Default hysteresis deadband width for anti-chattering mode stabilization ($\delta_{\mathrm{hyst}} = 0.15$).
pub const DEFAULT_HYSTERESIS_BAND: f32 = 0.15;

/// Sliding-window lookback horizon $M = 5$ for the Entropy Rate Integrator $\Phi_M(t) = \tilde{H}(t) - \tilde{H}(t-M)$.
pub const ENTROPY_GATE_SLIDING_WINDOW_HORIZON: usize = 5;

/// Capacity of the power-of-two ring buffer storing recent $\tilde{H}$ samples (`8`).
pub const ENTROPY_GATE_WINDOW_RING_CAPACITY: usize = 8;

/// Maximum barrier safety ceiling for numerical stabilization near zero lacuna scores ($10^9$).
pub const MAX_BARRIER_CEILING: f32 = 1.0e9;

/// Exponential moving average smoothing factor for Z-score entropy tracking ($\alpha_{\mathrm{EMA}} = 0.1$).
pub const EMA_ALPHA_ENTROPY: f32 = 0.1;

/// Upper ceiling for single-token Z-score inputs ($10.0\sigma$) preventing single-token outlier
/// spikes from locking `moving_z_entropy` into prolonged repulsion mode over subsequent tokens.
pub const MAX_VALID_TOKEN_Z_SCORE: f32 = DEFAULT_MAX_VALID_Z_SCORE;

/// Precomputed reciprocal of $\ln 2$ ($1 / \ln 2 \approx 1.44269504$) eliminating hardware `fdiv`
/// instructions during normalized entropy scaling.
pub(crate) const INV_LN_2: f32 = 1.0 / LN_2;

/// Constant Look-Up Table (LUT) for branch mode sample temperatures `[Mode 0, Mode 1, Mode 2]`.
pub(crate) const LUT_SAMPLE_TEMPERATURE: [f32; 3] = [1.0, 1.2, 1.5];

/// Constant Look-Up Table (LUT) for branch mode repulsion gain factors `[Mode 0, Mode 1, Mode 2]`.
pub(crate) const LUT_REPULSION_GAIN: [f32; 3] = [0.0, 0.5, 1.0];

pub mod controller;
pub mod kernels;

pub use self::{controller::*, kernels::*};

#[cfg(test)]
mod tests;
