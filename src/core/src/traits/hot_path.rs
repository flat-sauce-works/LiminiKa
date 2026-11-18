// name: src/core/src/traits/hot_path.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.2 Micro Observer traits (`traits/hot_path.rs`) for the Geometric Cellular Sheaf
//! Orchestrator (GCSO):
//! - **[`PointerActionHub`]**: $\mathcal{O}(1)$ 64-bit tagged pointer transitions and trail updates.
//! - **[`StigmergicChunkOps`]**: Mezzo PagedBlock chunk stepping.
//! - **[`HotPathExecutionEngine`] / [`FractalCascadeCoordinator`] / [`HotPathObserver`]**:
//!   Blanket-implemented micro observers over [`super::ActionHub`] with type-level
//!   [`BranchlessKernel`] enforcement.

#![deny(unsafe_code)]
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
#![allow(clippy::field_reassign_with_default)]

use super::aligned::AlignedSliceMut32;
use super::branchless::BranchlessKernel;
use super::primitives::{branchless_select_u8, GcsoResult};
use super::telemetry::{
    ContrastivePhaseBloomFilter, LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_PHASE_STEERED,
    LEDGER_METRIC_REPULSION_APPLIED,
};
use super::ActionHub;
use crate::abi::{gcso_ledger_slot_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t};

// ===================================================================
// 1. Action Hub Core Pointer Orchestrator Trait
// ===================================================================

/// Hot-path capability trait for $\mathcal{O}(1)$ tagged pointer transitions and trail updates.
///
/// Requires [`BranchlessKernel`] to enforce constant-time execution at the type level.
pub trait PointerActionHub: BranchlessKernel + Send + Sync {
    /// Resets transient pointer trails and slot counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_STATE`] if the Action Hub is locked or uninitialized.
    fn reset(&mut self) -> GcsoResult<()>;

    /// Advances a 64-bit tagged pointer transition in zero-allocation $\mathcal{O}(1)$ mode.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] or [`crate::abi::GCSO_ERROR_ACTION_HUB_FULL`] on failure.
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<()>;

    /// Updates cellular state for a trail directly with a Q7 phase delta in fast-path express mode.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `trail_id` exceeds Action Hub capacity.
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: gcso_q7_t,
    ) -> GcsoResult<()>;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `trail_id` is out of bounds or `pull_force` is non-finite.
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> GcsoResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if either trail ID is out of bounds.
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> GcsoResult<()>;

    /// Returns the maximum slot capacity of the Action Hub table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;
}

// ===================================================================
// 2. Mezzo Swarm Cell Chunk Operations
// ===================================================================

/// Mezzo-path trait for cellular swarm cell chunk step processing across PagedBlocks (**Mezzo Level**).
///
/// Requires [`BranchlessKernel`] to enforce constant-time execution at the type level.
pub trait StigmergicChunkOps: BranchlessKernel + Send + Sync {
    /// Updates local cellular swarm cell state across PagedBlock token chunk boundaries.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_INVALID_ARGUMENT`] if `chunk_len == 0`.
    fn step_chunk(&mut self, mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()>;
}

// ===================================================================
// 3. Lv.2 Micro Observers: `HotPathExecutionEngine`, `FractalCascadeCoordinator`, & `HotPathObserver`
// ===================================================================

/// Composite trait bundling all Hot-Path micro-capabilities registered on [`ActionHub`]
/// for monomorphic static dispatch with type-level [`BranchlessKernel`] enforcement.
pub trait HotPathExecutionEngine: ActionHub {}

impl<T: ActionHub + ?Sized> HotPathExecutionEngine for T {}

/// Blanket-implemented Lv.2 Micro-Observer coordinator that drives constant-time
/// $\mathcal{O}(1)$ `ActionHub -> FastPathLedger -> Bloom Repulsion -> QDPS/RIPA` token cascades.
pub trait FractalCascadeCoordinator: HotPathExecutionEngine {
    /// Executes a constant-time per-token stigmergic cascade over [`ActionHub`]:
    /// 1. Validates tensor dimensions, `GCSO_RESTRICT` disjointness, and radian thresholds.
    /// 2. Resolves `input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, steps the
    ///    pointer transition in `trail_out`, and applies [`ContrastivePhaseBloomFilter`] repulsion.
    /// 3. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR QDPS and steers
    ///    `query_tensor` and/or `key_tensor` with RIPA soft-clamping (`ripa_max_rad`).
    /// 4. Pushes the combined telemetry metric code (`LEDGER_METRIC_*`) into `ledger_slot`.
    ///
    /// Returns the observed Bloom repulsion bucket density (`0..=255`).
    ///
    /// # Errors
    /// Propagates any error from [`ActionHub::validate_hub_cascade_params`],
    /// [`ActionHub::step_hub_with_bloom_repulsion_to_words`], or
    /// [`super::telemetry::FastPathLedger::telemetry_push`].
    #[inline]
    fn execute_stigmergic_token_cascade(
        &mut self,
        input_ptr: u64,
        ledger_slot: &mut gcso_ledger_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_hub_cascade_params(
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, post_rep_words) = self
            .step_hub_with_bloom_repulsion_to_words(
                gcso_tagged_ptr_t(input_ptr),
                ledger_slot,
                bloom_filter,
                repulsion_threshold,
                repulsion_scale_u7,
                trail_out,
            )?;
        let steered = self.steer_qdps_words_prevalidated_core(
            &post_rep_words,
            head_dim,
            num_heads,
            qdps_min_step_rad,
            ripa_max_rad,
            query_tensor,
            key_tensor,
        );
        let metric = branchless_select_u8(bypass_hit, LEDGER_METRIC_FAST_PATH_HIT, 0)
            | branchless_select_u8(steered, LEDGER_METRIC_PHASE_STEERED, 0)
            | branchless_select_u8(repulsion_active, LEDGER_METRIC_REPULSION_APPLIED, 0);
        self.telemetry_push(ledger_slot, metric)?;
        Ok(density)
    }
}

impl<T: ActionHub + ?Sized> FractalCascadeCoordinator for T {}

/// Unified Lv.2 Micro Observer trait over [`ActionHub`] (`traits/hot_path.rs`).
pub trait HotPathObserver: HotPathExecutionEngine + FractalCascadeCoordinator {}

impl<T: ActionHub + ?Sized> HotPathObserver for T {}
