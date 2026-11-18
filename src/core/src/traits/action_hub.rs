// name: src/core/src/traits/action_hub.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core Composite Capability Ledger: [`ActionHub`] and zero-copy tensor / SWAR helpers.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
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
use super::branchless::{swar_words_nonzero_64, BranchlessKernel, Liminal};
use super::hot_path::PointerActionHub;
use super::primitives::{
    branchless_is_finite_f32, branchless_rad_to_qdps_step_u8, branchless_select_u32,
    branchless_select_u8, swar_q7_qdps_filter_words_64, GcsoResult, APERTURE_ACTION_HUB_WIDE,
    MAX_ATTENTION_HEADS,
};
use super::rope::{steer_from_swar_words_core_safe, PhaseSteering};
use super::telemetry::{
    ContrastivePhaseBloomFilter, FastPathLedger, LEDGER_METRIC_FAST_PATH_HIT,
    LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_REPULSION_APPLIED, NUM_STIGMERGIC_GRANULES,
    TRAIL_FLAG_REPULSION_ACTIVE,
};
use crate::abi::{gcso_ledger_slot_t, gcso_pointer_trail_t, gcso_tagged_ptr_t, GCSO_ERROR_INVALID_ARGUMENT};

// ===================================================================
// Core Composite Ledger: `ActionHub`
// ===================================================================

/// Validates optional Query/Key tensor slice lengths and `GCSO_RESTRICT` pairwise disjointness.
#[inline(always)]
pub(crate) fn validate_optional_qk_tensors(
    query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    expected_len: usize,
) -> GcsoResult<()> {
    match (query_tensor, key_tensor) {
        (Some(q), Some(k)) => {
            if q.len() < expected_len || k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            q.ensure_disjoint_mut(k)
        }
        (Some(q), None) => {
            if q.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, Some(k)) => {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, None) => Ok(()),
    }
}

/// Core Composite Ledger (**Central Trait Registry / Super-Registry**) uniting the branchless
/// foundation, RoPE steering kernels, $\mathcal{O}(1)$ pointer-trail transitions, and FastPathLedger
/// telemetry under static dispatch.
///
/// The Liminal Supertrait Blanket Extensions ([`super::gateway::ActionHubGatewayExt`]) observe this
/// composite directly.
pub trait ActionHub:
    BranchlessKernel + Liminal + PhaseSteering + PointerActionHub + FastPathLedger
{
    /// Broadest cognitive aperture spanned across all gateway, steering, swarm, cohomology, and
    /// mapped-storage capabilities (`0x0000_FFFF`).
    const HUB_APERTURE_MASK: u32 = APERTURE_ACTION_HUB_WIDE;

    /// Type-level complement identifying signals outside `ActionHub`'s horizon as unknowable (`!0xFFFF`).
    const HUB_UNKNOWABLE_VOID_MASK: u32 = !Self::HUB_APERTURE_MASK;

    /// Branchlessly partitions `signal_bits` into `(cognized_bits, unknowable_void_bits)` for
    /// [`Self::HUB_APERTURE_MASK`].
    #[inline(always)]
    fn inspect_hub_horizon_partition(&self, signal_bits: u32) -> (u32, u32) {
        self.partition_cognitive_and_unknowable_bits(Self::HUB_APERTURE_MASK, signal_bits)
    }

    /// Validates `head_dim`, `num_heads <= MAX_ATTENTION_HEADS`, `qdps_min_step_rad >= 0.0`,
    /// `ripa_max_rad > 0.0`, and optional Query/Key tensor lengths and `GCSO_RESTRICT` disjointness.
    ///
    /// Returns `(head_dim, num_heads, expected_tensor_len)` on success.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any dimension, radian threshold, or tensor slice
    /// invariant is violated.
    #[inline(always)]
    fn validate_hub_cascade_params(
        &self,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(usize, usize, usize)> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || qdps_min_step_rad < 0.0
            || !branchless_is_finite_f32(qdps_min_step_rad)
            || ripa_max_rad <= 0.0
            || !branchless_is_finite_f32(ripa_max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        validate_optional_qk_tensors(query_tensor, key_tensor, expected_len)?;
        Ok((head_dim, num_heads, expected_len))
    }

    /// Resolves `input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, advances the
    /// pointer transition in `trail_out`, evaluates `bloom_filter` anti-phase repulsion in-place,
    /// and returns `(bypass_hit, effective_ptr, repulsion_density, repulsion_active, post_repulsion_words)`
    /// directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Propagates any error returned by [`PointerActionHub::step_pointer`].
    #[inline(always)]
    fn step_hub_with_bloom_repulsion_to_words(
        &mut self,
        input_ptr: gcso_tagged_ptr_t,
        ledger_slot: &mut gcso_ledger_slot_t,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(
        bool,
        gcso_tagged_ptr_t,
        u8,
        bool,
        [u64; NUM_STIGMERGIC_GRANULES],
    )> {
        let (bypass_hit, effective_ptr) = self.resolve_or_pass_tagged_ptr(ledger_slot, input_ptr);
        self.step_pointer(effective_ptr.raw(), trail_out)?;

        let (density, repulsion_active, words) = bloom_filter
            .apply_phase_conjugate_repulsion_to_words(
                &mut trail_out.accumulated_phase_delta,
                repulsion_threshold,
                repulsion_scale_u7,
            );
        trail_out.flags |= branchless_select_u32(repulsion_active, TRAIL_FLAG_REPULSION_ACTIVE, 0);
        Ok((bypass_hit, effective_ptr, density, repulsion_active, words))
    }

    /// Filters `raw_words` via 8-lane SWAR QDPS and steers `query_tensor` / `key_tensor` using
    /// caller-validated `(head_dim, num_heads)`.
    #[inline(always)]
    fn steer_qdps_words_prevalidated_core(
        &self,
        raw_words: &[u64; NUM_STIGMERGIC_GRANULES],
        head_dim: usize,
        num_heads: usize,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> bool {
        let steered = query_tensor.is_some() | key_tensor.is_some();
        if !steered {
            return false;
        }
        let min_step_q7 = if qdps_min_step_rad.to_bits() == 0 {
            0
        } else {
            branchless_rad_to_qdps_step_u8(qdps_min_step_rad)
        };
        let filtered_buf;
        let swar_words = if min_step_q7 <= 1 {
            raw_words
        } else {
            filtered_buf = swar_q7_qdps_filter_words_64(raw_words, min_step_q7);
            &filtered_buf
        };
        if !swar_words_nonzero_64(swar_words) {
            return true;
        }

        match (query_tensor, key_tensor) {
            (Some(q), k_opt) => {
                steer_from_swar_words_core_safe(
                    swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    ripa_max_rad,
                    q,
                    k_opt,
                );
            }
            (None, Some(k)) => {
                steer_from_swar_words_core_safe(
                    swar_words,
                    0xFF,
                    128,
                    head_dim,
                    num_heads,
                    false,
                    ripa_max_rad,
                    k,
                    None,
                );
            }
            (None, None) => {}
        }
        true
    }

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
    /// [`FastPathLedger::telemetry_push`].
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

impl<T> ActionHub for T where
    T: BranchlessKernel + PhaseSteering + PointerActionHub + FastPathLedger + ?Sized
{
}
