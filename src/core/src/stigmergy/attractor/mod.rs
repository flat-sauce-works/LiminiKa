// name: src/core/src/stigmergy/attractor/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Attractor Classification Constants (`GcsoAnchorType`) and Blanket-Implemented
//! Episodic Sample -> RoPE Steering Coordination (`EpisodicPhaseSample`, `EpisodicSampleSteering`).

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

mod sample_steering;
pub use sample_steering::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        is_phase_profile_nonzero_64, AlignedBuffer32, PhaseSteering, DEFAULT_PHASE_CLAMP_MAX_RAD,
    };
    use crate::liminal::{gcso_q7_t, MAX_ATTENTION_HEADS};

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
    fn test_episodic_sample_steering_blanket_projections() {
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
    }
}
