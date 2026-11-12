// name: src/core/src/swarm.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete 64-byte cacheline-aligned [`SwarmEngine`] `Pod` state descriptor applying the
//! [`SwarmCellChunk`], [`HierarchicalTreeTable`], [`PagedBitmaskOps`], and [`StigmergicSwarmOps`]
//! traits from [`crate::traits`].

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

use crate::abi::{
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t,
    GCSO_ERROR_INVALID_ARGUMENT,
};
use crate::traits::primitives::MAX_ATTENTION_HEADS;
use crate::traits::{
    store_q7_words_64, GcsoResult, HierarchicalTreeTable, PagedBitmaskOps, Pod, StigmergicSwarmOps,
    SwarmCellChunk, SwarmResolutionReport, NUM_SWARM_MICRO_CELLS,
};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) state engine for the 8-cell SWAR Micro-Swarm
/// and QDPT/CPT hierarchical pointer cascade.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwarmEngine {
    /// 64-head Q7 phase profile partitioned into 8 Micro-Swarm cells of 8 heads each (`offset 0..64`).
    pub cell_phases: [gcso_q7_t; MAX_ATTENTION_HEADS],
}

// SAFETY: `SwarmEngine` is a 64-byte aligned array of 64 `i8` values with no padding traps.
unsafe impl Pod for SwarmEngine {}

impl Default for SwarmEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmEngine {
    /// Constructs a zero-initialized 64-byte [`SwarmEngine`].
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            cell_phases: [0; MAX_ATTENTION_HEADS],
        }
    }

    /// Constructs a 64-byte [`SwarmEngine`] initialized with `cell_phases`.
    #[inline(always)]
    #[must_use]
    pub const fn from_phases(cell_phases: [gcso_q7_t; MAX_ATTENTION_HEADS]) -> Self {
        Self { cell_phases }
    }

    /// Reinterprets the 64-byte phase buffer as 8 packed 64-bit SWAR cell words in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn words_ref(&self) -> &[u64; NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `SwarmEngine` is 64-byte aligned (`#[repr(C, align(64))]`) and `cell_phases` is at offset 0.
        unsafe {
            &*(self
                .cell_phases
                .as_ptr()
                .cast::<[u64; NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably reinterprets the 64-byte phase buffer as 8 packed 64-bit SWAR cell words in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    pub const fn words_mut(&mut self) -> &mut [u64; NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `SwarmEngine` is 64-byte aligned (`#[repr(C, align(64))]`) and `cell_phases` is at offset 0.
        unsafe {
            &mut *(self
                .cell_phases
                .as_mut_ptr()
                .cast::<[u64; NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Resolves `trail` against this engine's 8 Micro-Swarm cells via
    /// [`StigmergicSwarmOps::try_resolve_action_in_swarm_from_words`] and synchronizes
    /// `self.cell_phases` with the resolved SWAR phase words in $\mathcal{O}(1)$.
    #[inline(always)]
    pub fn resolve_trail_in_swarm(
        &mut self,
        trail: &mut gcso_pointer_trail_t,
        step_gain_u7: u8,
        qdps_step_q7: u8,
        tolerance_q7: u8,
        quorum_threshold: u8,
    ) -> SwarmResolutionReport {
        let target_words = *self.words_ref();
        let action_ptr = gcso_tagged_ptr_t(trail.current_ptr);
        let (report, next_words) = trail.try_resolve_action_in_swarm_from_words(
            action_ptr,
            &target_words,
            0xFF,
            step_gain_u7,
            qdps_step_q7,
            tolerance_q7,
            quorum_threshold,
            0.5,
            0.15,
            None,
            3,
            64,
        );
        store_q7_words_64(&mut self.cell_phases, &next_words);
        report
    }
}

impl SwarmCellChunk for SwarmEngine {
    #[inline(always)]
    fn step_chunk(&mut self, mask: &gcso_paged_bitmask_t, chunk_len: u32) -> GcsoResult<()> {
        if chunk_len == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let _active_bits = mask.popcount();
        Ok(())
    }
}

impl HierarchicalTreeTable for SwarmEngine {}

const _: () = {
    assert!(size_of::<SwarmEngine>() == 64);
    assert!(align_of::<SwarmEngine>() == 64);
    assert!(offset_of!(SwarmEngine, cell_phases) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::primitives::{ACTION_PRESET_EXPRESS_PHASE, ATTRACTOR_SLOT_PERSONA};
    use crate::traits::{ContrastivePhaseBloomFilter, StigmergicActionPtrOps};

    #[test]
    fn test_swarm_engine_bloom_and_bitmask_ops() {
        let mut engine = SwarmEngine::from_phases([24i8; MAX_ATTENTION_HEADS]);
        assert_eq!(engine.words_ref(), &[0x1818_1818_1818_1818u64; 8]);

        let mut mask = gcso_paged_bitmask_t::default();
        assert!(mask.is_all_zero());
        mask.set_bit_branchless(42);
        assert!(mask.test_bit_branchless(42));
        assert_eq!(mask.popcount(), 1);
        assert_eq!(engine.step_chunk(&mask, 64), Ok(()));

        let mut trail = gcso_pointer_trail_t {
            current_ptr: gcso_tagged_ptr_t::pack_action(
                0x1000,
                ATTRACTOR_SLOT_PERSONA,
                ACTION_PRESET_EXPRESS_PHASE,
                24,
            )
            .raw(),
            ..Default::default()
        };
        let report = engine.resolve_trail_in_swarm(&mut trail, 128, 1, 2, 4);
        assert_eq!(report.resolved_cell_count, 8);
        assert_eq!(report.hot_path_resolved, 1);
        assert_eq!(engine.cell_phases[0], 24);

        let mut bloom = ContrastivePhaseBloomFilter::new();
        let phases = [16i8; MAX_ATTENTION_HEADS];
        bloom.insert_repulsive_pattern(&phases);
        assert!(bloom.total_mass() > 0);
    }
}
