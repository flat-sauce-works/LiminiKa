// name: src/core/src/traits/hot_path.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.2 Micro-Capability traits (`traits/hot_path.rs`) for the Geometric Cellular Sheaf
//! Orchestrator (GCSO):
//! - **[`PointerActionHub`]**: $\mathcal{O}(1)$ 64-bit tagged pointer transitions and trail updates.
//! - **[`StigmergicChunkOps`]**: Mezzo PagedBlock chunk stepping.

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

use super::branchless::BranchlessKernel;
use super::primitives::GcsoResult;
use crate::abi::{gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t};

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

    /// Returns the active 8-bit lane enable mask (`0..=7`) on the working phase workbench (`0` = none).
    #[inline(always)]
    fn workbench_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 64-head SWAR words and non-negative lane weight for working-bank `lane` (`0..=7`)
    /// when enabled and non-zero.
    #[inline(always)]
    fn workbench_lane_words_and_weight(
        &self,
        _lane: u8,
    ) -> Option<([u64; crate::NUM_STIGMERGIC_GRANULES], f32)> {
        None
    }
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
