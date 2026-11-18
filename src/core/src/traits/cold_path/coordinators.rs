// name: src/core/src/traits/cold_path/coordinators.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cold-Path crystallization candidacy and full-stack blanket composite markers over [`ActionHub`].

use super::*;

// ===================================================================
// Blanket-Implemented Lv.3 Macro Observers & Full-Stack Composites
// ===================================================================

/// Cold-path crystallization candidacy extension trait for [`gcso_pointer_trail_t`].
pub trait ColdPathTrailCrystalExt: StigmergicTrailOps {
    /// Branchlessly evaluates whether a pointer trail qualifies for bottom-up macro
    /// crystallization (`stigmergic_density >= density_thresh && r_bar_t >= coherence_thresh && !repulsion_active`)
    /// and stamps or clears [`TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`] on `self.flags` in $\mathcal{O}(1)$.
    fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool;
}

impl ColdPathTrailCrystalExt for gcso_pointer_trail_t {
    #[inline(always)]
    fn evaluate_crystallization_candidacy(
        &mut self,
        density_crystallization_thresh: f32,
        coherence_crystallization_thresh: f32,
        r_bar_t: f32,
    ) -> bool {
        let density_ok = self.should_crystallize_branchless(density_crystallization_thresh)
            & branchless_is_finite_f32(density_crystallization_thresh);
        let coherence_ok = (r_bar_t >= coherence_crystallization_thresh)
            & (coherence_crystallization_thresh >= 0.0)
            & branchless_is_finite_f32(coherence_crystallization_thresh)
            & branchless_is_finite_f32(r_bar_t);
        let not_repulsed = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        let is_candidate = density_ok & coherence_ok & not_repulsed;
        self.flags = (self.flags & !TRAIL_FLAG_CRYSTALLIZED_CANDIDATE)
            | branchless_select_u32(is_candidate, TRAIL_FLAG_CRYSTALLIZED_CANDIDATE, 0);
        is_candidate
    }
}

/// Unified Lv.3 Macro Observer trait over [`ActionHub`] (`traits/cold_path.rs`).
pub trait ColdPathObserver:
    ActionHub
    + SheafCohomologyEvaluator
    + EdbcBranchController
    + AttractorField
    + MultiDomainAttractorSpace
    + PersonaPatcher
{
}

impl<T> ColdPathObserver for T where
    T: ActionHub
        + SheafCohomologyEvaluator
        + EdbcBranchController
        + AttractorField
        + MultiDomainAttractorSpace
        + PersonaPatcher
        + ?Sized
{
}

/// Composite Cold-Path Orchestrator trait uniting Macro evaluation, attractor steering, and mapped-storage persistence.
pub trait ColdPathOrchestrator:
    EntropyEvaluator + AttractorField + PersonaPatcher + L2pSvdProjector + KvKeyframeCache + MappedStorage
{
}

impl<T> ColdPathOrchestrator for T where
    T: EntropyEvaluator
        + AttractorField
        + PersonaPatcher
        + L2pSvdProjector
        + KvKeyframeCache
        + MappedStorage
        + ?Sized
{
}

/// Full-Stack Fractal GCSO Engine trait uniting Nano, Micro, Mezzo, and Macro contracts under static dispatch.
pub trait FullStackGcsoEngine:
    HotPathExecutionEngine
    + ColdPathOrchestrator
    + StigmergicChunkOps
    + GcsoRuntimeContext
    + SystemCapabilityQuery
{
}

impl<T> FullStackGcsoEngine for T where
    T: HotPathExecutionEngine
        + ColdPathOrchestrator
        + StigmergicChunkOps
        + GcsoRuntimeContext
        + SystemCapabilityQuery
        + ?Sized
{
}
