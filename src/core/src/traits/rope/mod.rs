// name: src/core/src/traits/rope/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lv.1 Capability 1: RoPE & Inline Tensor Modulation traits (`PhaseSteering`,
//! `EntropyGatedPhaseLookup`, `DynamicLayerSkipper`, `PspmRouter`, `SrlAdapter`,
//! and `VisualAttentionSteering`) registered into [`super::ActionHub`] for the
//! Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Key Optimizations & Architectural Guarantees
//! - **RIPA Upper-Quarter ($d_{\mathrm{head}}/4$) Low-Frequency Slicing**: All RIPA soft-clamped
//!   steering kernels (`apply_phase_steering_safe`, `apply_phase_steering_geodesic_safe`,
//!   `steer_from_swar_words_core_safe`) rotate strictly the upper low-frequency quarter
//!   `rot_dim = ((head_dim >> 2) & !1).max(2)` (`[base + head_dim - rot_dim .. base + head_dim]`),
//!   preserving lower high-frequency syntactic RoPE channels and achieving a `2.2x–3.3x` speedup
//!   over full-channel RoPE.
//! - **3-Tier SWAR Head-Run Coalescing, Branchless `TZCNT` Stride & 4K-Aliasing Avoidance**:
//!   Full-channel steering kernels (`apply_phase_steering_slice_core`,
//!   `apply_phase_steering_pair_slice_core`, `apply_phase_steering_batch`, and
//!   `apply_phase_steering_fused_batch`) combine: (1) multi-word uniform-phase coalescing with a
//!   direct whole-tensor fast path, (2) a 4-instruction SWAR 7-pair adjacent-inequality detector
//!   (`(adj.wrapping_sub(SWAR_SPLAT_BYTE_64) & !adj & SWAR_HIGH_BITS_7PAIRS_64) == 0`) that
//!   rotates 8 distinct-phase heads via a fixed-trip `chunks_exact_mut(head_dim)` loop with a
//!   single bounds check and zero inner branches, (3) hardware `TZCNT` sub-word run-length stride,
//!   and (4) sequential 32-float unrolled `Q`/`K` rotation (`apply_phase_steering_slice_with_sincos`) when
//!   `q_run.len() >= 128` (`512B`) to eliminate 2-stream 4KB page-offset Load-Store Queue aliasing.
//! - **Zero-Arithmetic Direct 256-Entry LUT Fast Path in PSPM Sub-Head Routing**: When
//!   `is_default_ripa && gain_u7 == 128 && pulse_q7 == 0`, `rotate_pspm_subgroup_with_slot_phases`
//!   and `rotate_fused_ripa_pspm_subgroup_words` select `Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF` or
//!   `Q7_DEFAULT_RIPA_SINCOS_REF` outside the head loop and index `lut[(slot_q7 as u8) as usize]`
//!   directly with zero per-head integer folding/scaling arithmetic and zero cache-branch jitter.
//! - **Single-Evaluation $SO(2)$ Group-Closure RIPA + PSPM Fusion**: Because both RIPA and PSPM
//!   operate on the upper low-frequency quarter `rot_dim`, `dispatch_fused_ripa_words_and_pspm_profiles_core`
//!   fuses $\mathbf{R}(\theta_{\mathrm{ripa}})\mathbf{R}(\theta_{\mathrm{pspm}}) = \mathbf{R}(\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}})$
//!   via `eval_fused_ripa_pspm_sincos_q7_fast` (evaluating the trigonometric polynomial once per
//!   head instead of twice on custom RIPA clamps) and caches consecutive head `(sin, cos)` pairs.
//! - **Zero-Jitter Micro-Swarm Quorum Pre-Pass & Constant-Time Damping**: `steer_from_swar_words_core_safe`
//!   resolves `quorum_mask` and `unresolved_damping_u7` in a branchless 8-word SWAR pre-pass, and
//!   [`PhaseSteering::apply_soft_phase_damping`] executes unconditional AVX2 subnormal-flushed scaling.

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
#![allow(unused_imports)]
use super::aligned::{
    AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut, AlignedSliceMut16,
    AlignedSliceMut32, DisjointMutPair, DisjointMutTriple,
};
use super::branchless::{BranchlessKernel, add_scalar_f32_slice_unrolled8, branchless_phase_coherence_inv_temp_scale, branchless_q7_coherence_and_resultant_64, branchless_q7_unified_field_fold_words_with_init_weight_64, clamp_angles_ripa_in_place_simd8, coherence_from_word_arrays_64, dot_f32_unrolled4, modulate_attention_scores_by_bayesian_trails_64, modulate_attention_scores_by_phase_bank_64, norm_l2_underflow_resilient_f32, procrustes_q7_slice_simd, rotate_so2_2d_xy_slice_simd, rotate_so2_3d_txy_slice_simd, rotate_so2_slice_unrolled16, scale_f32_slice_unrolled8, sincos_f32_x8_simd, srl_accum_outer_product_simd, srl_write_outer_product_simd};
use super::primitives::{DEFAULT_RIPA_Q7_TO_NORM_SCALE, F32_MIN_POSITIVE_NORMAL, Q7_DEFAULT_RIPA_SINCOS_REF, Q7_FULL_COS_I16_REF, Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF, Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF, Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF, Q7_SINCOS_LUT_256_REF, SWAR_SIGN_MASK_64, SWAR_SPLAT_BYTE_64, branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_inv_pos_f32, branchless_is_finite_f32, branchless_min_f32, branchless_q7_add_sat, branchless_q7_circular_diff, branchless_q7_fold_s1_geodesic, branchless_q7_scale_u7, branchless_qdps_filter_q7, branchless_rad_to_q7, branchless_rad_to_qdps_step_u8, branchless_relu_f32, branchless_rotate_so2_pair_scaled, branchless_select_f32, branchless_select_u32, branchless_select_u8, branchless_sincos_f32, branchless_sincos_q7, branchless_sincos_q7_ripa_preinv, branchless_splitmix64, eval_fused_ripa_pspm_sincos_q7_fast, eval_geodesic_ripa_sincos_q7_fast, eval_ripa_sincos_q7_fast, fast_inv_len_f32, fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, load_q7_x8_slice, sanitize_inplace_gain_f32, store_q7_words_64, store_q7_x8_slice, swar_add_sat_q7_slice_in_place, swar_circular_diff_and_fold_geodesic_q7_slices, swar_circular_diff_q7_slices, swar_fold_s1_geodesic_q7_slice_in_place, swar_q7_add_sat_x8, swar_q7_circular_add_x8, swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_qdps_filter_words_64, swar_q7_qdps_filter_x8, swar_q7_scale_le128_with_bias_x8, swar_q7_scale_le128_x8, swar_q7_scale_u7_x8, swar_qdps_filter_q7_slice_in_place, swar_sub_sat_q7_slices, swar_u8_sum_x8};
use super::primitives::{
    ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP, ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT,
    DEFAULT_RIPA_CLAMP_MAX_RAD, GcsoResult, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
    PAGED_BITMASK_BITS, Q7_PHASE_SCALE_RAD, unpack_action_mode_from_tag,
};
use crate::abi::{
    GCSO_ERROR_EDBC_SINGULARITY, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER,
    GCSO_ERROR_NULL_POINTER, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t,
    gcso_q7_t, gcso_srl_descriptor_t, gcso_tagged_ptr_t,
};
use core::f32::consts::{FRAC_PI_2, LN_2};
use core::mem::align_of;
use core::slice;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::branchless::{
    avx2_build_so2_2d_xy_vecs, avx2_hsum_ps, avx2_rotate_so2_2d_xy_slice_with_vecs,
    avx2_rotate_so2_block32, avx2_rotate_so2_pair_block32,
};

pub(crate) const NUM_SWARM_CELLS: usize = 8;
pub(crate) const HEADS_PER_CELL: usize = MAX_ATTENTION_HEADS / NUM_SWARM_CELLS;
const INV_Q14_SCALE_F32: f32 = 1.0 / 16384.0;


mod kernels;
pub use kernels::*;

mod phase_steering;
pub use phase_steering::*;

mod pspm;
pub use pspm::*;

mod adapters;
pub use adapters::*;

const _: () = {
    assert!(size_of::<gcso_pspm_config_t>() == 32);
    assert!(align_of::<gcso_pspm_config_t>() == 16);
    assert!(size_of::<gcso_srl_descriptor_t>() == 64);
    assert!(align_of::<gcso_srl_descriptor_t>() == 32);
    assert!(NUM_SWARM_CELLS * HEADS_PER_CELL == MAX_ATTENTION_HEADS);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::aligned::AlignedBuffer32;
    use crate::traits::primitives::DEFAULT_QDPS_MIN_STEP_RAD;

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
    impl EntropyGatedPhaseLookup for DummyRopeEngine {}
    impl DynamicLayerSkipper for DummyRopeEngine {}
    impl PspmRouter for DummyRopeEngine {}
    impl SrlAdapter for DummyRopeEngine {}
    impl VisualAttentionSteering for DummyRopeEngine {}

    #[test]
    fn test_phase_steering_lazy_unwrap_procrustes_slerp_and_bayesian_attention() {
        let eng = DummyRopeEngine {
            head_dim: 8,
            num_heads: 8,
        };
        let mut q_buf = AlignedBuffer32::new([1.0_f32; 64]);
        let mut k_buf = AlignedBuffer32::new([0.5_f32; 64]);
        let q_phases = AlignedBuffer16::new([40i8; 64]);
        let ctx_phases = AlignedBuffer16::new([16i8; 64]);

        let eff = eng
            .apply_seam_free_lazy_unwrap_q7(
                &mut q_buf.as_aligned_slice_mut(),
                Some(&mut k_buf.as_aligned_slice_mut()),
                &q_phases.as_aligned_slice(),
                &ctx_phases.as_aligned_slice(),
                DEFAULT_QDPS_MIN_STEP_RAD,
                Some(DEFAULT_RIPA_CLAMP_MAX_RAD),
            )
            .unwrap();
        assert_eq!(eff.data[0], 24);
        // Lower 3/4 high-frequency channels (0..6) are preserved; upper 1/4 (6..8) are RIPA-rotated:
        assert_eq!(&q_buf.data[0..6], &[1.0_f32; 6]);
        assert!((q_buf.data[6] - 1.0).abs() > 1.0e-3);

        let mut src_t = AlignedBuffer32::new([0.0_f32; 64]);
        for pair in src_t.data.chunks_exact_mut(2) {
            pair[0] = 1.0;
            pair[1] = 0.0;
        }
        let mut tgt_t = src_t;
        let p32 = AlignedBuffer16::new([32i8; 64]);
        eng.apply_phase_steering(&mut tgt_t.as_aligned_slice_mut(), &p32.as_aligned_slice())
            .unwrap();
        let mut proc_out = AlignedBuffer16::new([0i8; 64]);
        eng.compute_procrustes_phase_delta(
            &src_t.as_aligned_slice(),
            &tgt_t.as_aligned_slice(),
            &mut proc_out.as_aligned_slice_mut(),
        )
        .unwrap();
        assert_eq!(&proc_out.data[..8], &[32; 8]);

        let slerped = eng
            .slerp_and_apply_phase_steering_safe(
                &ctx_phases.as_aligned_slice(),
                &q_phases.as_aligned_slice(),
                64,
                DEFAULT_QDPS_MIN_STEP_RAD,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_buf.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert_eq!(slerped.data[0], 28);

        // Verify broadcast batch steering (single-token phase slice applied across 4 tokens)
        let mut q_batch = AlignedBuffer32::new([1.0_f32; 256]);
        eng.apply_phase_steering_batch(
            &mut q_batch.as_aligned_slice_mut(),
            &p32.as_aligned_slice(),
            4,
        )
        .unwrap();
        assert!((q_batch.data[0] - q_batch.data[192]).abs() < 1.0e-6);

        // Verify 3-tier SWAR dispatch (adjacent-distinct Tier 2 + TZCNT partial-run Tier 3)
        let mut p_het = AlignedBuffer16::new([0i8; 64]);
        let het_init: [i8; 8] = [8, -8, 16, -16, 24, 24, 0, 32];
        let mut h_idx = 0usize;
        while h_idx < 8 {
            p_het.data[h_idx] = het_init[h_idx];
            h_idx += 1;
        }
        let mut q_het = AlignedBuffer32::new([1.0_f32; 256]);
        let mut k_het = AlignedBuffer32::new([1.0_f32; 256]);
        eng.apply_phase_steering_fused_batch(
            &mut q_het.as_aligned_slice_mut(),
            &mut k_het.as_aligned_slice_mut(),
            &p_het.as_aligned_slice(),
            4,
        )
        .unwrap();
        assert!((q_het.data[0] - k_het.data[0]).abs() < 1.0e-6);
        assert!((q_het.data[0] - q_het.data[192]).abs() < 1.0e-6);

        // Verify exact binary dyadic reversible soft phase damping ([0.5, 2.0])
        let mut damp_buf = AlignedBuffer32::new([1.0_f32; 64]);
        eng.apply_soft_phase_damping(&mut damp_buf.as_aligned_slice_mut(), 0.5)
            .unwrap();
        eng.apply_soft_phase_damping(&mut damp_buf.as_aligned_slice_mut(), 2.0)
            .unwrap();
        assert_eq!(damp_buf.data[0].to_bits(), 1.0_f32.to_bits());

        let anchors = [[24i8; 64], [-96i8; 64]];
        let densities = [2.5_f32, 4.0_f32];
        let mut attn_scores = AlignedBuffer32::new([0.0_f32; 8]);
        let (mean_coh, best_idx) = eng
            .modulate_attention_scores_by_bayesian_trails(
                &mut attn_scores.as_aligned_slice_mut(),
                &[24i8; 64],
                &anchors,
                &densities,
                Some(0),
                2.0,
                0.5,
                0.25,
            )
            .unwrap();
        assert!(mean_coh.is_finite());
        assert_eq!(best_idx, 0);
        assert!(attn_scores.data[0] > attn_scores.data[1]);
    }

    #[test]
    fn test_autonomous_swarm_frequency_aperture_steering() {
        let eng16 = DummyRopeEngine {
            head_dim: 16,
            num_heads: 4,
        };
        assert_eq!(eng16.resolve_autonomous_rot_dim(0x02, 0xFF, 0.1, 2.0), 2);
        assert_eq!(eng16.resolve_autonomous_rot_dim(0x00, 0x0F, 0.1, 1.0), 4);
        assert_eq!(eng16.resolve_autonomous_rot_dim(0x04, 0xFF, 0.2, 1.5), 6);
        assert_eq!(eng16.resolve_autonomous_rot_dim(0x08, 0xFF, 0.6, 1.0), 8);

        let q_phases = AlignedBuffer16::new([40i8; 64]);
        let mut q_auto = AlignedBuffer32::new([1.0_f32; 64]);
        let used_rot = eng16
            .apply_autonomous_swarm_phase_steering(
                &mut q_auto.as_aligned_slice_mut(),
                None,
                &q_phases.as_aligned_slice(),
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                0x04,
                0xFF,
                0.2,
                1.5,
            )
            .unwrap();
        assert_eq!(used_rot, 6);
        assert_eq!(&q_auto.data[0..10], &[1.0_f32; 10]);
        assert!((q_auto.data[10] - 1.0).abs() > 1.0e-3);
    }
}
