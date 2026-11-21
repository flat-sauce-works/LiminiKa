// name: src/core/src/stigmergy/attractor/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Macro Attractor Field Engine, Dynamic 8-Lane Working Space, Blanket-Implemented
//! Episodic Sample -> RoPE Steering / Persona Growth Coordinator, and `.gcso` `Pod` Snapshot
//! Persistence for the Geometric Cellular Sheaf Orchestrator (GCSO).

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(non_camel_case_types)]
#![allow(clippy::must_use_candidate)]

/// Classification types for topological attractor field anchors (Section 5.3).
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum GcsoAnchorType {
    /// Natural language system prompt anchor projected into phase space (Primary Endpoint).
    SystemPrompt = 0,
    /// Dense feature embedding vector anchor in latent space.
    Embedding = 1,
    /// Phase-conjugate anti-phase repulsion anchor ($-\boldsymbol{\Delta\theta}$).
    PhaseRepulse = 2,
    /// Manifold topological knot/tree anchor.
    Topological = 3,
    /// Self-organized macro anchor crystallized bottom-up from high-density pointer trails.
    Crystallized = 4,
}

impl GcsoAnchorType {
    /// Converts a raw `u32` value to a [`GcsoAnchorType`] enum variant if valid.
    #[inline(always)]
    #[must_use]
    pub const fn from_u32(val: u32) -> Option<Self> {
        match val {
            0 => Some(Self::SystemPrompt),
            1 => Some(Self::Embedding),
            2 => Some(Self::PhaseRepulse),
            3 => Some(Self::Topological),
            4 => Some(Self::Crystallized),
            _ => None,
        }
    }
}

/// C-ABI type alias for anchor classification (`gcso_anchor_type_t`).
pub type gcso_anchor_type_t = u32;
/// System prompt string anchor descriptor type (`0`).
pub const GCSO_ANCHOR_TYPE_SYSTEM_PROMPT: gcso_anchor_type_t = 0;
/// Feature embedding anchor descriptor type (`1`).
pub const GCSO_ANCHOR_TYPE_EMBEDDING: gcso_anchor_type_t = 1;
/// Phase repulsion anchor descriptor type (`2`).
pub const GCSO_ANCHOR_TYPE_PHASE_REPULSE: gcso_anchor_type_t = 2;
/// Topological knot anchor descriptor type (`3`).
pub const GCSO_ANCHOR_TYPE_TOPOLOGICAL: gcso_anchor_type_t = 3;
/// Crystallized invariant anchor descriptor type (`4`).
pub const GCSO_ANCHOR_TYPE_CRYSTALLIZED: gcso_anchor_type_t = 4;

// ===================================================================

pub use crate::liminal::MAX_ATTRACTOR_DOMAINS;

/// Fixed capacity of topological anchors per Macro Attractor Field accumulator (`64`).
pub const MAX_ANCHORS_PER_FIELD: usize = 64;

/// Binary magic constant for `.gcso` snapshot headers (`0x4F534347` = ASCII `"GCSO"`).
pub const GCSO_SNAPSHOT_MAGIC_U32: u32 = 0x4F53_4347;

const _: () = {
    assert!(MAX_ANCHORS_PER_FIELD == 64);
};

/// Maximum dimension supported for embedding vectors in inline zero-alloc storage (`128`).
pub const MAX_EMBEDDING_DIM: usize = 128;

pub(super) const SNAPSHOT_HEADER_BYTES: usize = 64;

mod sample_steering;
pub use sample_steering::*;

mod snapshot;
pub use snapshot::*;

mod field_engine;
pub use field_engine::*;

mod multi_space;
pub use multi_space::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        is_phase_profile_nonzero_64, AlignedBuffer32, AlignedBuffer64, PhaseSteering,
        DEFAULT_PHASE_CLAMP_MAX_RAD,
    };
    use crate::liminal::{gcso_q7_t, GCSO_ERROR_ACTION_HUB_FULL, MAX_ATTENTION_HEADS};
    use crate::stigmergy::trail::gcso_pointer_trail_t;
    use core::mem::size_of;

    struct DummyPhaseSteerer;
    impl PhaseSteering for DummyPhaseSteerer {
        fn head_dim(&self) -> u32 {
            8
        }
        fn num_heads(&self) -> u32 {
            2
        }
    }

    #[test]
    fn test_attractor_anchor_energy_and_autonomous_phase_proximity_gravity() {
        let mut anchor_near = AttractorAnchor {
            is_active: 1,
            weight: 1.0,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            last_active_step: 0,
            half_life_steps: 100,
            phase_deltas: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let anchor_opposed = AttractorAnchor {
            phase_deltas: [-96i8; MAX_ATTENTION_HEADS],
            ..anchor_near
        };

        assert!((anchor_near.compute_energy_score(0) - 2.0).abs() < 1e-5);
        assert!((anchor_near.compute_energy_score(100) - 1.0).abs() < 1e-5);

        let trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let (pull_near, coh_near) = anchor_near.compute_resonant_pull(0, &trail);
        let (pull_opp, coh_opp) = anchor_opposed.compute_resonant_pull(0, &trail);
        assert!((coh_near - 1.0).abs() < 1e-4);
        assert!((coh_opp - (-1.0)).abs() < 1e-4);
        assert!(pull_near > pull_opp * 7.0);

        anchor_near.touch(150, 0.5);
        assert_eq!(anchor_near.last_active_step, 150);
        assert!((anchor_near.stigmergic_density - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_accumulator_pressure_evaporative_displacement_and_s1_geodesic_force() {
        let mut engine = AttractorFieldEngine::new(0, 3);
        let buf = AlignedBuffer32::new([0.5f32; 16]);
        let slice = buf.as_aligned_slice();

        let id1 = engine.add_system_prompt_anchor("Anchor 1", 1.0).unwrap();
        let id2 = engine
            .add_embedding_anchor(&slice, GCSO_ANCHOR_TYPE_EMBEDDING, 0.5)
            .unwrap();
        let _id3 = engine.add_system_prompt_anchor("Anchor 3", 2.0).unwrap();

        assert_eq!(engine.active_count(), 3);
        assert!(engine.is_saturated());
        assert_eq!(
            engine.add_system_prompt_anchor("Weak Anchor", 0.01),
            Err(GCSO_ERROR_ACTION_HUB_FULL)
        );

        let id4 = engine
            .add_system_prompt_anchor("Strong Anchor 4", 5.0)
            .unwrap();
        assert_eq!(engine.active_count(), 3);
        assert!(engine.get_anchor(id1).is_some());
        assert!(engine.get_anchor(id2).is_none());
        assert!(engine.get_anchor(id4).is_some());

        engine.anchors[0].stigmergic_density = 3.0;
        assert_eq!(engine.aggregate_bottom_up().unwrap(), 1);
        assert_eq!(engine.crystallized_count(), 1);

        let evaporated = engine.evaporate_weakest_anchor();
        assert!(evaporated.is_some());
        assert_eq!(engine.active_count(), 2);
    }

    #[test]
    fn test_coherence_safe_blanket_steering_and_gcso_pod_snapshot_roundtrip() {
        let mut space = MultiAttractorSpace::new();
        let emb_buf = AlignedBuffer32::new([0.25f32; 16]);
        let emb_slice = emb_buf.as_aligned_slice();

        for _ in 0..80 {
            assert!(space.register_persona_patch_anchor(&emb_slice, 1.0).is_ok());
        }
        assert_eq!(space.total_unified_anchor_count(), 80);
        assert_eq!(space.persona_field().active_count(), 64);

        let mut trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let mut bank = [[0i8; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS];
        let mut lane_pulls = [0.0f32; MAX_ATTRACTOR_DOMAINS];
        let total_pull = space.project_active_field_bank_8(&trail, &mut bank, &mut lane_pulls);
        assert!(total_pull > 0.0);
        assert_eq!(
            bank[usize::from(crate::stigmergy::ATTRACTOR_SLOT_PERSONA)][0],
            32
        );

        let weights = space.rebalance_weights_from_trail(&trail);
        assert!(weights[usize::from(crate::stigmergy::ATTRACTOR_SLOT_PERSONA)] >= 1.0);

        let (syn_pull, _coh, neighbors) = space
            .steer_trail_by_coupled_neighborhood(0.5, 0.2, &mut trail)
            .unwrap();
        assert!(syn_pull > 0.0);
        assert!(neighbors >= 1);

        // 1. Verify Blanket EpisodicSampleSteering across diverse sample types (&str, &[u8], [i8; 64], [f32; N], tuples)
        let steerer = DummyPhaseSteerer;
        let mut q_tensor = AlignedBuffer32::new([1.0f32; 16]);
        let str_deltas = "persistent persona anchor sample"
            .steer_rope_tensors_direct(
                &steerer,
                1,
                0.01,
                DEFAULT_PHASE_CLAMP_MAX_RAD,
                &mut q_tensor.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert!(is_phase_profile_nonzero_64(&str_deltas));

        let raw_q7_sample: [gcso_q7_t; MAX_ATTENTION_HEADS] = [32i8; MAX_ATTENTION_HEADS];
        let q7_deltas = raw_q7_sample.to_rope_phase_deltas(1, 0.01);
        assert_eq!(q7_deltas[0], 32);

        let f32_sample = [0.25f32; 16];
        let f32_deltas = f32_sample.to_rope_phase_deltas(2, 0.01);
        assert_eq!(f32_deltas[0], 32);

        // 2. Verify AttractorGrowthCoordinator batch absorption & concentration growth (kappa_c)
        space.reset_all_lanes();
        let mut growth_trail = gcso_pointer_trail_t::default();
        let mut persona_profile = [0i8; MAX_ATTENTION_HEADS];

        let episodes = [
            ([28i8; MAX_ATTENTION_HEADS], 1.5_f32),
            ([30i8; MAX_ATTENTION_HEADS], 2.0_f32),
            ([32i8; MAX_ATTENTION_HEADS], 2.5_f32),
        ];
        let (report, crystallized_cnt) = space
            .absorb_episodic_sample_batch(
                &episodes,
                1,
                0.8,
                0.7,
                &mut growth_trail,
                &mut persona_profile,
            )
            .unwrap();
        assert!(crystallized_cnt >= 1);
        assert!(report.resultant_r_bar > 0.95);
        assert!(report.von_mises_kappa > 10.0);
        assert!((28..=32).contains(&persona_profile[0]));

        let mut step_rope_deltas = [0i8; MAX_ATTENTION_HEADS];
        let (str_rep, beta_eff) = space
            .absorb_and_steer_rope_step(
                "episodic memory anchor sample",
                1,
                0.5,
                0.5,
                0.01,
                1.0,
                &mut growth_trail,
                &mut step_rope_deltas,
            )
            .unwrap();
        assert!(str_rep.resonant_pull_force > 0.0);
        assert!(beta_eff > 1.0);

        // 3. Verify .gcso Pod Snapshot (AttractorGrowthSnapshot) serialization & zero-copy restoration
        let expected_unified_count = space.total_unified_anchor_count();
        let mut snap_buf =
            AlignedBuffer64::<u8, { size_of::<AttractorGrowthSnapshot>() }>::zeroed();
        let written = space
            .serialize_growth_snapshot_to_bytes(&growth_trail, snap_buf.as_mut_slice(), 0)
            .unwrap();
        assert_eq!(written, size_of::<AttractorGrowthSnapshot>());

        space.reset_all_lanes();
        let zero_copy_snap = space
            .restore_from_snapshot_bytes_zero_copy(snap_buf.as_slice(), 0)
            .unwrap();
        assert_eq!(zero_copy_snap.magic, GCSO_SNAPSHOT_MAGIC_U32);
        assert!(zero_copy_snap.active_anchor_count >= 1);
        assert!(zero_copy_snap.composite_kappa > 0.0);
        assert_eq!(space.total_unified_anchor_count(), expected_unified_count);
    }
}
