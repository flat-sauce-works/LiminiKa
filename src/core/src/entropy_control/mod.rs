// name: src/core/src/entropy_control.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Entropy-Driven Decoding Branch Controller (EDBC) core engine implementation.
//!
//! # Architectural Role (Thin `Pod` State Composite + Blanket Closed-Loop & Gateway Coordinator)
//! - **[`EntropyController`]**: Compact 128-byte (2-cacheline, `#[repr(C, align(32))]`) [`Pod`] stateful
//!   controller managing Moving Z-Score Normalized Attention Entropy $\tilde{H}$, the 8-slot
//!   Sliding-Window Entropy Rate Integrator $\Phi_M(t)$ ($M = 5$), anti-chattering dwell-step
//!   inertia, and Langevin pitchfork branch mode transitions (`Mode 0/1/2`). Directly implements
//!   **Origin 1 (`CausalProvenanceSignal` + `SystemConstraintSignal`)** and [`ManifoldGatewaySignal`]
//!   so `&EntropyController` can be used both standalone and inside `(S, F, P)` / `(S, F, P, G)` tuples.
//! - **[`EntropyClosedLoopCoordinator`]**: Blanket-implemented extension trait
//!   (`impl<T: EdbcBranchController + SheafCohomologyEvaluator + HierarchicalTreeTable + ?Sized> EntropyClosedLoopCoordinator for T {}`)
//!   that equips [`EntropyController`], [`crate::traits::ColdPathObserver`], and
//!   [`crate::traits::FullStackGcsoEngine`] with:
//!   1. Single-pass AVX2 $\mathcal{C}_{\mathrm{void}}$ + $\ell_1 / \ell_2$ coherence-ratio evaluation
//!      (`eval_liminal_void_with_coherence_ratio`) and strictly branchless `0-ln` Taylor-Estrin
//!      surrogate-to-Z-score stepping (`eval_surrogate_and_step`).
//!   2. Single-pass register-chained Limina inter-memory trail-chain cohomology verification under
//!      the Joint Probability Chain Rule $\sum_t \ln P(M_t \mid M_{t-1})$ with deterministic
//!      repulsion pointer override (`step_with_liminal_trail_chain_and_bloom_guard`).
//!   3. Register-resident Single-Manifold Unified-Field EDBC pointer evolution
//!      (`step_with_unified_field_and_pointer`), preserving 32-bit `target_anchor_id` in `cluster_id`.
//!   4. Single-scan closed-loop **Pitchfork Metastable Tunneling + Bayesian Pointer-Trail selection,
//!      rewiring, Limina bridging & Attention score modulation**
//!      (`step_with_bayesian_pointer_selection_and_tunneling`,
//!      `route_coarse_to_fine_and_step_bayesian_tunneling`), combining sequential conditional
//!      transition probability ($\sum_t \ln P(M_t \mid M_{t-1})$ $\times$ Attention phase interference)
//!      with indirect pointer-chain routing in `ActionHub`.
//!   5. Loop-hoisted zero-copy candidate-trail Bayes-Attention score modulation coupled with EDBC
//!      energy-guided inverse-temperature scaling (`modulate_candidate_trails_with_edbc_temperature`).
//!   6. 256-bit dyadic tree QDPT depth resolution and pointer cascading
//!      (`resolve_dyadic_tree_and_step_edbc`).
//!   7. **2-Origin Composite Manifold Gateway, $\mathcal{O}(1)$ Causal Provenance, Online Toroidal
//!      $k$-Means / DP-Means Macro Clustering & `.gcso` Sample-Driven Attractor Crystallization**
//!      (`step_via_gateway_frame`, `step_via_gateway_signal`, `step_via_gateway_kmeans_64`,
//!      `step_with_crystallized_ensemble_and_pointer`).

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

use core::f32::consts::{FRAC_PI_2, LN_2};
use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t,
    GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::aligned::{AlignedSlice32, AlignedSliceMut32};
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
use crate::traits::branchless::coherence_from_word_arrays_with_both_uniform_flags_64;
use crate::traits::branchless::{
    branchless_fast_transition_log_prob, branchless_phase_coherence_inv_temp_scale,
    branchless_q7_unified_field_fold_words_with_init_weight_64, scale_f32_slice_unrolled8,
    sum_abs_and_sq_f32_simd8, sum_sq_f32_simd8, swar_words_equal_64,
};
use crate::traits::cold_path::{
    EdbcBranchController, EntropyEvaluator, HierarchicalTreeTable, SheafCohomologyEvaluator,
};
use crate::traits::gateway::{
    synthesize_multi_origin_gateway_frame, synthesize_multi_origin_gateway_frame_with_words,
    CausalProvenanceReport, CausalProvenanceSignal, ContradictionGuard,
    CrystallizedAttractorRecord, GatewayFrame, ManifoldGatewaySignal, ObjectiveFact,
    SubjectivePersona, SystemConstraintSignal, CAUSAL_AXIS_SYSTEM_CONSTRAINT,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, CAUSAL_CAUSE_COHOMOLOGY_REPULSION,
    CAUSAL_CAUSE_SYSTEM_QOS_THROTTLE, DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
    ROOT_CAUSE_COHOMOLOGY_REPULSION, ROOT_CAUSE_NOMINAL, ROOT_CAUSE_SYSTEM_QOS_THROTTLE,
};
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::traits::primitives::Q7_FULL_COS_REF;
use crate::traits::primitives::{
    branchless_abs_f32, branchless_clamp_f32, branchless_eyring_kramers_escape,
    branchless_flush_subnormal_f32, branchless_is_finite_f32, branchless_max_f32,
    branchless_min_f32, branchless_popcount_rsqrt_u32, branchless_q7_add_sat, branchless_rad_to_q7,
    branchless_relu_f32, branchless_rsqrt_f32, branchless_sanitize_z_score_f32,
    branchless_select_f32, branchless_select_q7, branchless_select_u32, branchless_select_u64,
    branchless_select_u8, branchless_splitmix64, branchless_sqrt_f32, fast_inv_len_f32,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, store_q7_word_at,
    store_q7_words_64, swar_q7_add_sat_x8, swar_q7_neg_sat_x8, swar_q7_qdps_filter_words_64,
    tanh_lut_eval_nonneg, trail_phase_words_mut, trail_phase_words_ref, DEFAULT_MAX_VALID_Z_SCORE,
    F32_MIN_POSITIVE_NORMAL, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    GcsoResult, Pod, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_PRESET_INERTIAL_SKIP,
    ACTION_PRESET_REPULSION_ESCAPE, ATTRACTOR_SLOT_ID_MASK, ATTRACTOR_SLOT_REPULSION,
    DEFAULT_ENTROPY_EPS, MAX_ATTENTION_HEADS, MAX_SURROGATE_ENTROPY_LN2,
};
use crate::traits::telemetry::{
    select_and_bridge_macro_aligned_bayesian_trail_core, ContrastivePhaseBloomFilter,
    StigmergicActionPtrOps, StigmergicTrailOps, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    NUM_STIGMERGIC_GRANULES, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, TRAIL_FLAG_LIMINAL_BEAT_ACTIVE,
    TRAIL_FLAG_REPULSION_ACTIVE, TRAIL_FLAG_SYNERGY_RESONANCE,
};
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
