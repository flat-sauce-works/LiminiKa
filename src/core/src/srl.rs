// name: src/core/src/srl.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Concrete 64-byte cacheline-aligned [`SrlEngine`] `Pod` state descriptor applying the
//! [`PspmRouter`], [`SrlAdapter`], [`VisualAttentionSteering`], [`PersonaPatcher`], and
//! [`L2pSvdProjector`] traits from [`crate::traits`].

#![forbid(unsafe_op_in_unsafe_fn)]
#![deny(missing_docs)]
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

use crate::abi::{gcso_srl_descriptor_t, GCSO_ERROR_INVALID_ARGUMENT};
use crate::traits::branchless::{
    branchless_clamp_f32, branchless_is_finite_f32, circular_r_bar_from_words_64,
};
use crate::traits::gateway::project_bytes_swar_to_words_64;
use crate::traits::{
    GcsoResult, L2pSvdProjector, PersonaPatcher, Pod, PspmRouter, SrlAdapter,
    VisualAttentionSteering,
};
use core::mem::{align_of, offset_of, size_of};

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Synchronous Residual Layer (SRL) Rank-1 outer product adapters, PSPM sub-head routing,
/// Visual 2D/3D RoPE steering, and L2P-SVDLoRA basis projection.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SrlEngine {
    /// Embedded 64-byte C-ABI v0.1.1 SRL Rank-1 descriptor (`offset 0..64`).
    pub descriptor: gcso_srl_descriptor_t,
}

// SAFETY: `SrlEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `gcso_srl_descriptor_t`.
unsafe impl Pod for SrlEngine {}

impl Default for SrlEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(1.0, 0, 1)
    }
}

impl SrlEngine {
    /// Constructs a new 64-byte aligned [`SrlEngine`] with canonical C-ABI v0.1.1 descriptor fields.
    #[inline(always)]
    #[must_use]
    pub const fn new(scale_factor: f32, layer_idx: u32, rank: u32) -> Self {
        Self {
            descriptor: gcso_srl_descriptor_t {
                layer_idx,
                rank: if rank == 0 { 1 } else { rank },
                u_vector_ptr: 0,
                v_vector_ptr: 0,
                gain_scalar_ptr: 0,
                scale_factor,
                flags: 0,
                reserved: [0; 24],
            },
        }
    }

    /// Validates the embedded `gcso_srl_descriptor_t` fields (`rank > 0` and finite positive `scale_factor`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `rank == 0` or `scale_factor` is non-positive/non-finite.
    #[inline(always)]
    pub const fn validate(&self) -> GcsoResult<()> {
        if self.descriptor.rank == 0
            || self.descriptor.scale_factor <= 0.0
            || !branchless_is_finite_f32(self.descriptor.scale_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

impl PspmRouter for SrlEngine {}
impl SrlAdapter for SrlEngine {}
impl VisualAttentionSteering for SrlEngine {}
impl L2pSvdProjector for SrlEngine {}
impl PersonaPatcher for SrlEngine {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
        if patch_data.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let words = project_bytes_swar_to_words_64(patch_data);
        let r_bar = circular_r_bar_from_words_64(&words);
        self.descriptor.scale_factor = branchless_clamp_f32(0.5 + r_bar, 0.25, 2.0);
        Ok(())
    }
}

const _: () = {
    assert!(size_of::<SrlEngine>() == 64);
    assert!(align_of::<SrlEngine>() == 64);
    assert!(offset_of!(SrlEngine, descriptor) == 0);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{gcso_paged_bitmask_t, gcso_pspm_config_t, gcso_q7_t, gcso_tagged_ptr_t};
    use crate::traits::aligned::{AlignedBuffer32, AlignedBuffer64};
    use crate::traits::primitives::{
        ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
        DEFAULT_RIPA_CLAMP_MAX_RAD,
    };
    use crate::traits::rope::{DynamicLayerSkipper, PhaseSteering};

    struct DummyRopeEngine {
        head_dim: u32,
        num_heads: u32,
    }
    impl PhaseSteering for DummyRopeEngine {
        fn head_dim(&self) -> u32 {
            self.head_dim
        }
        fn num_heads(&self) -> u32 {
            self.num_heads
        }
    }
    impl DynamicLayerSkipper for DummyRopeEngine {}
    impl PspmRouter for DummyRopeEngine {}
    impl SrlAdapter for DummyRopeEngine {}
    impl VisualAttentionSteering for DummyRopeEngine {}

    #[test]
    fn test_srl_engine_pod_and_persona_patch() {
        let mut engine = SrlEngine::default();
        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(engine.apply_patch(b"LiminiKa-SRL-Patch"), Ok(()));
        assert!(engine.descriptor.scale_factor >= 0.25);
    }

    #[test]
    fn test_pspm_srl_layer_skipper_and_visual_steering() {
        let eng = DummyRopeEngine {
            head_dim: 12,
            num_heads: 8,
        };
        let pspm_cfg = gcso_pspm_config_t {
            num_fact_heads: 2,
            num_logic_heads: 4,
            num_explore_heads: 2,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.2,
            ..Default::default()
        };
        let mut q_pspm = AlignedBuffer32::new([1.0_f32; 96]);
        let bank = AlignedBuffer64::<gcso_q7_t, 512>::splat(24);
        eng.dispatch_ripa_and_subhead_slot_ensemble_fused(
            &mut q_pspm.as_aligned_slice_mut(),
            &[16i8; 64],
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            &pspm_cfg,
            &bank.relax_slice::<16>(),
            ATTRACTOR_SLOT_KNOWLEDGE,
            ATTRACTOR_SLOT_PERSONA,
            2,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            8,
            12,
            true,
        )
        .unwrap();
        assert_eq!(&q_pspm.data[0..10], &[1.0_f32; 10]);
        assert!((q_pspm.data[10] - 1.0).abs() > 1.0e-3);

        let mut y_srl = AlignedBuffer32::new([1.0_f32; 8]);
        let x_srl = AlignedBuffer32::new([1.0_f32; 8]);
        let u_srl = AlignedBuffer32::new([0.5_f32; 8]);
        let v_srl = AlignedBuffer32::new([0.25_f32; 8]);
        let g_srl = AlignedBuffer32::new([2.0_f32; 8]);
        let desc = gcso_srl_descriptor_t {
            layer_idx: 0,
            rank: 1,
            u_vector_ptr: u_srl.as_aligned_slice().as_ptr() as u64,
            v_vector_ptr: v_srl.as_aligned_slice().as_ptr() as u64,
            gain_scalar_ptr: 0,
            scale_factor: 1.0,
            flags: 0,
            reserved: [0; 24],
        };
        let ptr = gcso_tagged_ptr_t::pack(
            0x1000,
            (ATTRACTOR_SLOT_PERSONA << 5) | ACTION_PRESET_FULL_ORCHESTRATION,
            0,
        );
        let injected = eng
            .eval_rank1_pointer_gated_fused(
                ptr,
                &mut y_srl.as_aligned_slice_mut(),
                &x_srl.as_aligned_slice(),
                &desc,
            )
            .unwrap();
        assert!(injected);
        assert!((y_srl.data[0] - 2.0).abs() < 1.0e-5);

        let desc_with_gain = gcso_srl_descriptor_t {
            gain_scalar_ptr: g_srl.as_aligned_slice().as_ptr() as u64,
            ..desc
        };
        eng.eval_rank1_fused(
            &mut y_srl.as_aligned_slice_mut(),
            &x_srl.as_aligned_slice(),
            &desc_with_gain,
        )
        .unwrap();
        assert!((y_srl.data[0] - 4.0).abs() < 1.0e-5);

        let mut frame_cur = AlignedBuffer32::new([1.0_f32; 48]);
        let frame_prv = AlignedBuffer32::new([1.0_f32; 48]);
        frame_cur.data[0..12].fill(2.0);
        let mut mask = gcso_paged_bitmask_t::default();
        let active_patches = eng
            .detect_frame_mse_bitmask(
                &frame_cur.as_aligned_slice(),
                &frame_prv.as_aligned_slice(),
                4,
                12,
                0.1,
                &mut mask,
            )
            .unwrap();
        assert_eq!(active_patches, 1);
    }
}
