// name: src/core/src/attractor/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Macro Attractor Field Engine, Dynamic 8-Lane Working Space, Blanket-Implemented
//! Episodic Sample -> RoPE Steering / Persona Growth Coordinator, and `.gcso` `Pod` Snapshot
//! Persistence for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Architectural Role (Unified Phase Manifold, `0-fdiv` $S^1$ Geodesic Gravity & `.gcso` Growth)
//! Rather than partitioning latent memory into rigid, manually engineered feature boxes,
//! this module models all persona traits, world lore, episodic dialogue turns, and anti-hallucination
//! repulsion traces as sample points (anchors) coexisting on a **single continuous phase manifold**
//! ($SO(2)^{d_{\mathrm{head}}/2}$, represented as 64-byte Q7 phase vectors):
//!
//! 1. **Coherence-Safe Base Signal Trait (`EpisodicPhaseSample`)**:
//!    Any cognitive input (`str`, `&str`, `[u8]`, `&[u8]`, `[u8; N]`, `[gcso_q7_t; 64]`, `[f32]`,
//!    `[f32; N]`, `AlignedSlice`, `gcso_pointer_trail_t`, `gcso_tagged_ptr_t`, `AttractorAnchor`,
//!    `CrystallizedAnchorRecord`, and weighted tuples `(S, f32)` / `(S, f32, u32)`) implements
//!    [`EpisodicPhaseSample`] via strictly disjoint head types and reference/tuple blanket delegators,
//!    completely avoiding upstream `AsRef<[u8]>` coherence collisions (`E0119`).
//! 2. **Blanket-Implemented RoPE Steering & Attractor Growth (`EpisodicSampleSteering` & `AttractorGrowthCoordinator`)**:
//!    - [`EpisodicSampleSteering`] (`impl<T: EpisodicPhaseSample + ?Sized>`) grants every sample type
//!      automatic conversion to SWAR-scaled, QDPS-filtered RoPE phase deltas (`[gcso_q7_t; 64]`),
//!      direct in-place Query/Key tensor RIPA phase steering, constant-time resonance absorption into
//!      [`gcso_pointer_trail_t`], and direct crystallization into [`AttractorAnchor`].
//!    - [`AttractorGrowthCoordinator`] (`impl<T: AttractorField + PersonaPatcher + ?Sized>`) plots incoming
//!      samples onto the manifold. As sample count grows, circular resultant length $\bar{R}_c \to 1$
//!      and von Mises concentration $\kappa_c$ rise, sharpening gravitational pull and inverse-temperature
//!      scaling $\beta_{\mathrm{eff}}$ before bottom-up crystallizing into persistent anchors.
//! 3. **Zero-Copy `.gcso` `Pod` Snapshot Persistence (`CrystallizedAnchorRecord` & `AttractorGrowthSnapshot`)**:
//!    Crystallized sample points, synthesized persona phase profiles, and concentration $\kappa_c$
//!    are packed into an 8,448-byte (8.25 KB) 64-byte-aligned [`Pod`] snapshot (`AttractorGrowthSnapshot`)
//!    with branchless IEEE 802.3 CRC32 verification for zero-copy ZIMMS `.gcso` distribution.

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

use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_anchor_type_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t, GCSO_ABI_VERSION_HEX,
    GCSO_ANCHOR_TYPE_CRYSTALLIZED, GCSO_ANCHOR_TYPE_EMBEDDING, GCSO_ANCHOR_TYPE_PHASE_REPULSE,
    GCSO_ANCHOR_TYPE_SYSTEM_PROMPT, GCSO_ANCHOR_TYPE_TOPOLOGICAL, GCSO_ERROR_ACTION_HUB_FULL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_VERSION_MISMATCH,
};
use crate::traits::aligned::{
    AlignedBuffer16, AlignedBuffer32, AlignedSlice, AlignedSlice16, AlignedSlice32,
    AlignedSliceMut32,
};
use crate::traits::branchless::{
    branchless_phase_coherence_inv_temp_scale, branchless_q7_inhibit_words_64,
    branchless_q7_resonant_pull_64, branchless_q7_synergy_fuse_words_64,
    circular_r_bar_from_words_64, circular_resultant_from_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, swar_words_nonzero_64,
};
use crate::traits::cold_path::{
    AttractorField, ColdPathTrailCrystalExt, MultiDomainAttractorSpace, PersonaPatcher,
};
use crate::traits::gateway::{
    CausalProvenanceSignal, CognitiveGroundingSignal, CrystallizedAttractorRecord, GatewayFrame,
    ManifoldGatewaySignal, SubjectivePersonaSignal, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, ROOT_CAUSE_NOMINAL,
};
use crate::traits::primitives::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32, branchless_q7_conjugate_64,
    branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_select_f32,
    branchless_select_i32, branchless_select_u32, branchless_select_u8, branchless_splitmix64,
    branchless_von_mises_kappa, crc32_block64_zero_seed_words, crc32_step_block64_words,
    crc32_step_l16, crc32_step_l32, crc32_step_l8, fast_inv_len_f32, fast_inv_pos_normal_f32,
    is_phase_profile_nonzero_64, is_word_array_uniform_64, load_q7_words_64, store_q7_word_at,
    store_q7_words_64, swar_q7_qdps_filter_words_64, swar_q7_scale_le128_with_bias_x8,
    swar_q7_sub_sat_x8, trail_phase_words_ref, Q7_FULL_COS_REF, SWAR_LOW7_MASK_64,
    SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64,
};
use crate::traits::primitives::{
    is_slot_pair_coupled, GcsoResult, Pod, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    GCSO_SNAPSHOT_MAGIC_U32, MAX_ATTENTION_HEADS,
};
use crate::traits::rope::PhaseSteering;
use crate::traits::telemetry::{
    StigmergicActionPtrOps, StigmergicTrailOps, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    TRAIL_FLAG_REPULSION_ACTIVE,
};

pub use crate::traits::primitives::{MAX_ANCHORS_PER_FIELD, MAX_ATTRACTOR_DOMAINS};

/// Maximum dimension supported for embedding vectors in inline zero-alloc storage (`128`).
pub const MAX_EMBEDDING_DIM: usize = 128;

/// Default attractor lane identifier for System Baseline Constraints (`0`).
pub const DOMAIN_SYSTEM_BASELINE: u32 = 0;
/// Default attractor lane identifier for Persona Patches (`1`).
pub const DOMAIN_PERSONA: u32 = 1;
/// Default attractor lane identifier for Generalized Knowledge Sources (`2`).
pub const DOMAIN_KNOWLEDGE: u32 = 2;
/// Default attractor lane identifier for Anti-Hallucination Phase Repulsion (`3`).
pub const DOMAIN_PHASE_REPULSION: u32 = 3;
/// Default attractor lane identifier for User Interaction Context (`4`).
pub const DOMAIN_USER_INTERACTION: u32 = 4;
/// Dynamic extension working lane identifier 5 (`Visual / Screen Co-Watching`).
pub const DOMAIN_CUSTOM_5: u32 = 5;
/// Dynamic extension working lane identifier 6 (`Mood / Circadian Drift`).
pub const DOMAIN_CUSTOM_6: u32 = 6;
/// Dynamic extension working lane identifier 7 (`Workshop Extension / Overflow`).
pub const DOMAIN_CUSTOM_7: u32 = 7;

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
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::DEFAULT_RIPA_CLAMP_MAX_RAD;

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
        assert_eq!(bank[DOMAIN_PERSONA as usize][0], 32);

        let weights = space.rebalance_weights_from_trail(&trail);
        assert!(weights[DOMAIN_PERSONA as usize] >= 1.0);

        let (syn_pull, _coh, neighbors) = space
            .steer_trail_by_coupled_neighborhood(0.5, 0.2, &mut trail)
            .unwrap();
        assert!(syn_pull > 0.0);
        assert!(neighbors >= 1);

        // 1. Verify Blanket EpisodicSampleSteering across diverse sample types (&str, &[u8], [i8; 64], [f32; N], tuples)
        let steerer = DummyPhaseSteerer;
        let mut q_tensor = AlignedBuffer32::new([1.0f32; 16]);
        let str_deltas = "Monika literature club persona"
            .steer_rope_tensors_direct(
                &steerer,
                1,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
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
                "Monika literature club memory",
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
