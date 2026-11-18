// name: src/core/src/entropy_control.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-Driven Decoding Branch Controller (EDBC) core engine implementation.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Surrogate Entropy Coordinator)
//! - **[`EntropyController`]**: Compact 128-byte (2-cacheline, `#[repr(C, align(32))]`) [`Pod`] stateful
//!   controller managing Moving Z-Score Normalized Attention Entropy $\tilde{H}$, the 8-slot
//!   Sliding-Window Entropy Rate Integrator $\Phi_M(t)$ ($M = 5$), anti-chattering dwell-step
//!   inertia, and Langevin pitchfork branch mode transitions (`Mode 0/1/2`). Directly implements
//!   **Origin 1 (`CausalProvenanceSignal` + `SystemConstraintSignal`)** and [`ManifoldGatewaySignal`]
//!   so `&EntropyController` can be used both standalone and inside `(S, F, P)` / `(S, F, P, G)` tuples.
//! - **[`EntropyClosedLoopCoordinator`]**: Blanket-implemented extension trait
//!   (`impl<T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized> EntropyClosedLoopCoordinator for T {}`)
//!   providing branchless `0-ln` Taylor-Estrin Top-2 surrogate-to-Z-score stepping
//!   (`eval_surrogate_and_step`).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
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

use core::f32::consts::LN_2;
use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_edbc_state_t, gcso_tagged_ptr_t, GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::aligned::AlignedSlice32;
use crate::traits::branchless::sum_sq_f32_simd8;
use crate::traits::cold_path::{
    EdbcBranchController, EntropyEvaluator, HierarchicalTreeTable, SheafCohomologyEvaluator,
};
use crate::traits::gateway::{
    synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
    CausalProvenanceSignal, ContradictionGuard, GatewayFrame, ManifoldGatewaySignal, ObjectiveFact,
    SubjectivePersona, SystemConstraintSignal, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
    CAUSAL_CAUSE_COHOMOLOGY_REPULSION, CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE,
};
use crate::traits::primitives::{
    branchless_clamp_f32, branchless_eyring_kramers_escape, branchless_flush_subnormal_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32, branchless_relu_f32,
    branchless_sanitize_z_score_f32, branchless_select_f32, branchless_select_u32,
    branchless_select_u8, fast_inv_len_f32, fast_inv_pos_normal_f32, tanh_lut_eval_nonneg,
    DEFAULT_MAX_VALID_Z_SCORE, F32_MIN_POSITIVE_NORMAL,
};
use crate::traits::primitives::{
    GcsoResult, Pod, ATTRACTOR_SLOT_ID_MASK, DEFAULT_ENTROPY_EPS, MAX_SURROGATE_ENTROPY_LN2,
};
use crate::traits::telemetry::{StigmergicActionPtrOps, NUM_STIGMERGIC_GRANULES};
/// Normal entropy convergence mode (`Mode 0`: single stable attractor at origin $x^* = 0$).
pub const EDBC_MODE_NORMAL_ENTROPY: u32 = 0;

/// Pitchfork bifurcation exploration mode (`Mode 1`: supercritical double-well $\pm x^*_\pm$).
pub const EDBC_MODE_PITCHFORK_BIFURCATION: u32 = 1;

/// Phase-conjugate repulsion mode (`Mode 2`: anti-phase $-\boldsymbol{\Delta\theta}$ escape).
pub const EDBC_MODE_PHASE_CONJUGATE_REPULSION: u32 = 2;

/// Default logarithmic singularity epsilon fallback constant ($10^{-12}$, `c_abi_spec.md`).
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
pub const EDBC_SLIDING_WINDOW_HORIZON: usize = 5;

/// Capacity of the power-of-two ring buffer storing recent $\tilde{H}$ samples (`8`).
pub const EDBC_WINDOW_RING_CAPACITY: usize = 8;

/// Maximum barrier safety ceiling for numerical stabilization near zero void scores ($10^9$).
pub const MAX_BARRIER_CEILING: f32 = 1.0e9;

/// Exponential moving average smoothing factor for Z-score entropy tracking ($\alpha_{\mathrm{EMA}} = 0.1$).
pub const EMA_ALPHA_ENTROPY: f32 = 0.1;

/// Upper ceiling for single-token Z-score inputs ($10.0\sigma$) preventing single-token outlier
/// spikes from locking `moving_z_entropy` into prolonged repulsion mode over subsequent tokens.
pub const MAX_VALID_TOKEN_Z_SCORE: f32 = DEFAULT_MAX_VALID_Z_SCORE;

/// Precomputed reciprocal of $\ln 2$ ($1 / \ln 2 \approx 1.44269504$) eliminating hardware `fdiv`
/// instructions during normalized entropy scaling.
pub(super) const INV_LN_2: f32 = 1.0 / LN_2;

/// Constant Look-Up Table (LUT) for branch mode sample temperatures `[Mode 0, Mode 1, Mode 2]`.
pub(super) const LUT_SAMPLE_TEMPERATURE: [f32; 3] = [1.0, 1.2, 1.5];

/// Constant Look-Up Table (LUT) for branch mode repulsion gain factors `[Mode 0, Mode 1, Mode 2]`.
pub(super) const LUT_REPULSION_GAIN: [f32; 3] = [0.0, 0.5, 1.0];

pub mod closed_loop;
pub mod controller;
pub mod kernels;

pub use self::{closed_loop::*, controller::*, kernels::*};

#[cfg(test)]
mod tests;
