// name: src/core/src/stigmergy/trail/traits.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Capability traits [`PointerSubstrate`] and [`StigmergicSubstrateOps`].

use super::PointerTrail;
use crate::foundation::{
    AlignedSliceMut32, TaggedPtr, branchless_is_finite_f32, branchless_rad_to_phase_step_u8,
    steer_from_swar_words_core_safe, swar_q7_phase_step_filter_words_64, swar_words_nonzero_64,
};
use crate::liminal::{
    BitCut32, LMNK_ERR_INVALID_ARGUMENT, LmnkResult, MAX_ATTENTION_HEADS, NUM_STIGMERGIC_GRANULES,
    PhaseQ7, THRESHOLD_STIGMERGY_WIDE, branchless_select_u8, branchless_select_u32,
};
use crate::stigmergy::ledger::{ContrastivePhaseBloomFilter, FastPathLedger, LedgerSlot};
use crate::stigmergy::{
    LEDGER_METRIC_FAST_PATH_HIT, LEDGER_METRIC_PHASE_STEERED, LEDGER_METRIC_REPULSION_APPLIED,
    TRAIL_FLAG_REPULSION_ACTIVE,
};

/// Hot-path capability trait for $\mathcal{O}(1)$ tagged pointer transitions and trail updates.
pub trait PointerSubstrate: Send + Sync {
    /// Resets transient pointer trails and slot counters without freeing pre-allocated tables.
    ///
    /// # Errors
    /// Returns [`crate::liminal::LMNK_ERR_INVALID_STATE`] if the substrate is locked or uninitialized.
    fn reset(&mut self) -> LmnkResult<()>;

    /// Advances a 64-bit tagged pointer transition in zero-allocation $\mathcal{O}(1)$ mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] or [`crate::liminal::LMNK_ERR_SUBSTRATE_FULL`] on failure.
    fn step_pointer(&mut self, current_ptr: u64, trail_out: &mut PointerTrail) -> LmnkResult<()>;

    /// Updates cellular state for a trail directly with a Q7 phase delta in fast-path express mode.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_id` exceeds substrate capacity.
    fn update_cellular_state_express(
        &mut self,
        trail_id: u32,
        phase_delta: PhaseQ7,
    ) -> LmnkResult<()>;

    /// Pulls active pointer trails toward a macro target anchor via pull force $F_{\mathrm{pull}}$.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `trail_id` is out of bounds or `pull_force` is non-finite.
    fn pull_trail_to_attractor(
        &mut self,
        trail_id: u32,
        anchor_id: u32,
        pull_force: f32,
    ) -> LmnkResult<()>;

    /// Links adjacent cellular hallucinated trails into contiguous stigmergic trace graphs.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if either trail ID is out of bounds.
    fn link_hallucinated_trails(&mut self, src_trail_id: u32, dst_trail_id: u32) -> LmnkResult<()>;

    /// Returns the maximum slot capacity of the substrate table.
    fn capacity(&self) -> u32;

    /// Returns the active pointer trail count currently registered.
    fn active_count(&self) -> u32;

    /// Returns the active 8-bit lane enable mask (`0..=7`) on the working phase bank (`0` = none).
    #[inline(always)]
    fn lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) tagged with anti-phase repulsion behavior.
    #[inline(always)]
    fn repulsion_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) tagged as isolated exterior-only sandbox hypotheses.
    #[inline(always)]
    fn sandbox_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 8-bit bitmask of lanes (`0..=7`) explicitly pinned against auto-eviction or release.
    #[inline(always)]
    fn pinned_lane_mask(&self) -> u8 {
        0
    }

    /// Returns the 64-head SWAR words and non-negative lane weight for working-bank `lane` (`0..=7`)
    /// when enabled and non-zero.
    #[inline(always)]
    fn lane_words_and_weight(&self, _lane: u8) -> Option<([u64; NUM_STIGMERGIC_GRANULES], f32)> {
        None
    }
}

/// Validates optional Query/Key tensor slice lengths and `LMNK_RESTRICT` pairwise disjointness.
#[inline(always)]
pub(crate) fn validate_optional_qk_tensors(
    query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    expected_len: usize,
) -> LmnkResult<()> {
    match (query_tensor, key_tensor) {
        (Some(q), Some(k)) => {
            if q.len() < expected_len || k.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            q.ensure_disjoint_mut(k)
        }
        (Some(q), None) => {
            if q.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
            }
            Ok(())
        }
        (None, Some(k)) => {
            if k.len() < expected_len {
                return Err(LMNK_ERR_INVALID_ARGUMENT);
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
/// Blanket extensions (`SubstrateGatewayExt`) observe this composite directly.
pub trait StigmergicSubstrateOps: PointerSubstrate + FastPathLedger {
    /// Broadest cognitive aperture spanned across all gateway, steering, swarm, cohomology, and
    /// mapped-storage capabilities (`0x0000_FFFF`).
    const HUB_APERTURE_MASK: u32 = THRESHOLD_STIGMERGY_WIDE;

    /// Type-level complement identifying signals outside `StigmergicSubstrateOps`'s horizon as unobserved (`!0xFFFF`).
    const HUB_UNOBSERVED_LACUNA_MASK: u32 = !Self::HUB_APERTURE_MASK;

    /// Branchlessly partitions `signal_bits` into `(settled_bits, unobserved_lacuna_bits)` for
    /// [`Self::HUB_APERTURE_MASK`].
    #[inline(always)]
    fn inspect_substrate_horizon_partition(&self, signal_bits: u32) -> (u32, u32) {
        let cut = BitCut32::split(signal_bits, Self::HUB_APERTURE_MASK);
        (cut.inside, cut.outside)
    }

    /// Validates `head_dim`, `num_heads <= MAX_ATTENTION_HEADS`, `phase_step_min_rad >= 0.0`,
    /// `max_rad > 0.0`, and optional Query/Key tensor lengths and `LMNK_RESTRICT` disjointness.
    ///
    /// Returns `(head_dim, num_heads, expected_tensor_len)` on success.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any dimension, radian threshold, or tensor slice
    /// invariant is violated.
    #[inline(always)]
    fn validate_substrate_cascade_params(
        &self,
        head_dim: u32,
        num_heads: u32,
        phase_step_min_rad: f32,
        max_rad: f32,
        query_tensor: Option<&AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<(usize, usize, usize)> {
        let valid_dims = (head_dim != 0)
            & ((head_dim & 1) == 0)
            & (num_heads != 0)
            & ((num_heads as usize) <= MAX_ATTENTION_HEADS);
        if !valid_dims
            || phase_step_min_rad < 0.0
            || !branchless_is_finite_f32(phase_step_min_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let hd = head_dim as usize;
        let nh = num_heads as usize;
        let expected_len = hd.checked_mul(nh).ok_or(LMNK_ERR_INVALID_ARGUMENT)?;
        validate_optional_qk_tensors(query_tensor, key_tensor, expected_len)?;
        Ok((hd, nh, expected_len))
    }

    /// Resolves `input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, advances the
    /// pointer transition in `trail_out`, evaluates `bloom_filter` anti-phase repulsion in-place,
    /// and returns `(bypass_hit, effective_ptr, repulsion_density, repulsion_active, post_repulsion_words)`
    /// directly in 8 `u64` SWAR registers.
    ///
    /// # Errors
    /// Propagates any error returned by [`PointerSubstrate::step_pointer`].
    #[inline(always)]
    fn step_substrate_with_bloom_repulsion_to_words(
        &mut self,
        input_ptr: TaggedPtr,
        ledger_slot: &mut LedgerSlot,
        bloom_filter: &ContrastivePhaseBloomFilter,
        repulsion_threshold: u8,
        repulsion_scale_u7: u8,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<(bool, TaggedPtr, u8, bool, [u64; NUM_STIGMERGIC_GRANULES])> {
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

    /// Filters `raw_words` via 8-lane SWAR `PhaseStepFilter` and steers `query_tensor` / `key_tensor` using
    /// caller-validated `params`.
    #[inline(always)]
    fn steer_phase_step_words_prevalidated_core(
        &self,
        raw_words: &[u64; NUM_STIGMERGIC_GRANULES],
        params: PhaseStepSteerParams,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> bool {
        let PhaseStepSteerParams {
            head_dim,
            num_heads,
            phase_step_min_rad,
            max_rad,
        } = params;
        let steered = query_tensor.is_some() | key_tensor.is_some();
        if !steered {
            return false;
        }
        let min_step_q7 = if phase_step_min_rad.to_bits() == 0 {
            0
        } else {
            branchless_rad_to_phase_step_u8(phase_step_min_rad)
        };
        let filtered_buf;
        let swar_words = if min_step_q7 <= 1 {
            raw_words
        } else {
            filtered_buf = swar_q7_phase_step_filter_words_64(raw_words, min_step_q7);
            &filtered_buf
        };
        if !swar_words_nonzero_64(swar_words) {
            return true;
        }

        match (query_tensor, key_tensor) {
            (Some(q), k_opt) => {
                steer_from_swar_words_core_safe(
                    swar_words, 0xFF, 128, head_dim, num_heads, false, max_rad, q, k_opt,
                );
            }
            (None, Some(k)) => {
                steer_from_swar_words_core_safe(
                    swar_words, 0xFF, 128, head_dim, num_heads, false, max_rad, k, None,
                );
            }
            (None, None) => {}
        }
        true
    }

    /// Executes a constant-time per-token stigmergic cascade over [`StigmergicSubstrateOps`]:
    /// 1. Validates tensor dimensions, `LMNK_RESTRICT` disjointness, and radian thresholds.
    /// 2. Resolves `params.input_ptr` against `ledger_slot`'s $\mathcal{O}(1)$ shortcut table, steps the
    ///    pointer transition in `trail_out`, and applies [`ContrastivePhaseBloomFilter`] repulsion.
    /// 3. Filters `trail_out.accumulated_phase_delta` via 8-lane SWAR `PhaseStepFilter` and steers
    ///    `query_tensor` and/or `key_tensor` with `PhaseClamp` soft-clamping (`params.max_rad`).
    /// 4. Pushes the combined telemetry metric code (`LEDGER_METRIC_*`) into `ledger_slot`.
    ///
    /// Returns the observed Bloom repulsion bucket density (`0..=255`).
    ///
    /// # Errors
    /// Propagates any error from [`StigmergicSubstrateOps::validate_substrate_cascade_params`],
    /// [`StigmergicSubstrateOps::step_substrate_with_bloom_repulsion_to_words`], or
    /// [`FastPathLedger::telemetry_push`].
    #[inline]
    fn execute_stigmergic_token_cascade(
        &mut self,
        params: TokenCascadeParams<'_>,
        ledger_slot: &mut LedgerSlot,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: &mut PointerTrail,
    ) -> LmnkResult<u8> {
        let (head_dim, num_heads, _expected_len) = self.validate_substrate_cascade_params(
            params.head_dim,
            params.num_heads,
            params.phase_step_min_rad,
            params.max_rad,
            query_tensor.as_deref(),
            key_tensor.as_deref(),
        )?;
        let (bypass_hit, _eff_ptr, density, repulsion_active, post_rep_words) = self
            .step_substrate_with_bloom_repulsion_to_words(
                TaggedPtr(params.input_ptr),
                ledger_slot,
                params.bloom_filter,
                params.repulsion_threshold,
                params.repulsion_scale_u7,
                trail_out,
            )?;
        let steered = self.steer_phase_step_words_prevalidated_core(
            &post_rep_words,
            PhaseStepSteerParams {
                head_dim,
                num_heads,
                phase_step_min_rad: params.phase_step_min_rad,
                max_rad: params.max_rad,
            },
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

/// Pre-validated phase-step filtering and RoPE steering parameters for
/// [`StigmergicSubstrateOps::steer_phase_step_words_prevalidated_core`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseStepSteerParams {
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Minimum phase-step filter threshold in radians.
    pub phase_step_min_rad: f32,
    /// Maximum phase clamp in radians.
    pub max_rad: f32,
}

/// Parameters for [`StigmergicSubstrateOps::execute_stigmergic_token_cascade`].
#[derive(Debug, Clone, Copy)]
pub struct TokenCascadeParams<'a> {
    /// Raw 64-bit tagged pointer input.
    pub input_ptr: u64,
    /// Contrastive phase Bloom filter for repulsion detection.
    pub bloom_filter: &'a ContrastivePhaseBloomFilter,
    /// Attention head dimension.
    pub head_dim: u32,
    /// Number of attention heads.
    pub num_heads: u32,
    /// Bloom repulsion density threshold (`0..=255`).
    pub repulsion_threshold: u8,
    /// Repulsion conjugate blend scale (`0..=128`).
    pub repulsion_scale_u7: u8,
    /// Minimum phase-step filter threshold in radians.
    pub phase_step_min_rad: f32,
    /// Maximum phase clamp in radians.
    pub max_rad: f32,
}

impl<T> StigmergicSubstrateOps for T where T: PointerSubstrate + FastPathLedger + ?Sized {}
