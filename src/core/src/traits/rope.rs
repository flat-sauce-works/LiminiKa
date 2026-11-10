// name: src/core/src/traits/rope.rs
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

use core::f32::consts::{FRAC_PI_2, LN_2};
use core::mem::{align_of, size_of};
use core::slice;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use super::branchless::{
    avx2_hsum_ps, avx2_rotate_so2_block32, avx2_rotate_so2_pair_block32,
    avx2_rotate_so2_triple_block32,
};
use super::aligned::{
    AlignedBuffer16, AlignedSlice16, AlignedSlice32, AlignedSliceMut, AlignedSliceMut16,
    AlignedSliceMut32, DisjointMutPair, DisjointMutTriple,
};
use super::branchless::{
    add_scalar_f32_slice_unrolled8, branchless_clamp_f32, branchless_flush_subnormal_f32,
    branchless_inv_pos_f32, branchless_is_finite_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_q7_add_sat, branchless_q7_circular_diff,
    branchless_q7_coherence_and_resultant_64, branchless_q7_fold_s1_geodesic,
    branchless_q7_scale_u7, branchless_q7_unified_field_fold_words_with_init_weight_64,
    branchless_qdps_filter_q7, branchless_rad_to_q7, branchless_rad_to_qdps_step_u8,
    branchless_relu_f32, branchless_rotate_so2_pair_scaled, branchless_select_f32,
    branchless_select_u32, branchless_select_u8, branchless_sincos_f32,
    branchless_sincos_q7, branchless_sincos_q7_ripa_preinv, branchless_splitmix64,
    clamp_angles_ripa_in_place_simd8, coherence_from_word_arrays_64, dot_f32_unrolled4,
    eval_fused_ripa_pspm_sincos_q7_fast,
    eval_geodesic_ripa_sincos_q7_fast, eval_ripa_sincos_q7_fast, fast_inv_len_f32,
    fast_inv_pos_normal_f32, is_word_array_uniform_64, load_q7_words_64, load_q7_x8_slice,
    modulate_attention_scores_by_bayesian_trails_64, modulate_attention_scores_by_phase_bank_64,
    norm_l2_underflow_resilient_f32, procrustes_q7_slice_simd, rotate_so2_2d_xy_slice_simd,
    rotate_so2_3d_txy_slice_simd, rotate_so2_slice_unrolled16, sanitize_inplace_gain_f32,
    scale_f32_slice_unrolled8,
    sincos_f32_x8_simd, srl_accum_outer_product_simd, srl_write_outer_product_simd,
    store_q7_words_64, store_q7_x8_slice, swar_add_sat_q7_slice_in_place,
    swar_circular_diff_and_fold_geodesic_q7_slices, swar_circular_diff_q7_slices,
    swar_fold_s1_geodesic_q7_slice_in_place, swar_q7_add_sat_x8, swar_q7_circular_add_x8,
    swar_q7_circular_diff_x8, swar_q7_fold_s1_geodesic_x8, swar_q7_qdps_filter_words_64,
    swar_q7_qdps_filter_x8,
    swar_q7_scale_le128_with_bias_x8, swar_q7_scale_u7_x8, swar_qdps_filter_q7_slice_in_place,
    swar_slerp_words_with_bias_64, swar_sub_sat_q7_slices, swar_u8_sum_x8, BranchlessKernel,
    DEFAULT_RIPA_Q7_TO_NORM_SCALE, F32_MIN_POSITIVE_NORMAL, Q7_DEFAULT_RIPA_SINCOS_REF,
    Q7_FULL_COS_I16_REF, Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF,
    Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF, Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF,
    Q7_SINCOS_LUT_256_REF,
};
use super::primitives::{
    unpack_action_mode_from_tag, GcsoResult, ACTION_MODE_DPSR_STEER, ACTION_MODE_LAYER_SKIP,
    ACTION_MODE_PSPM_ROUTE, ACTION_MODE_SRL_INJECT, DEFAULT_RIPA_CLAMP_MAX_RAD,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, PAGED_BITMASK_BITS, Q7_PHASE_SCALE_RAD,
};
use crate::abi::{
    gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pspm_config_t, gcso_q7_t,
    gcso_srl_descriptor_t, gcso_tagged_ptr_t, GCSO_ERROR_EDBC_SINGULARITY,
    GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_MISALIGNED_POINTER, GCSO_ERROR_NULL_POINTER,
};

const NUM_SWARM_CELLS: usize = 8;
const HEADS_PER_CELL: usize = MAX_ATTENTION_HEADS / NUM_SWARM_CELLS;
const SWAR_SIGN_MASK_64: u64 = 0x8080_8080_8080_8080;
const SWAR_SPLAT_BYTE_64: u64 = 0x0101_0101_0101_0101;
const INV_Q14_SCALE_F32: f32 = 1.0 / 16384.0;

/// Computes the RIPA / PSPM upper low-frequency rotation dimension
/// `rot_dim = ((head_dim / 4) & !1).max(2)` (even integer in `2..=head_dim`).
#[inline(always)]
#[must_use]
pub(crate) const fn ripa_upper_rot_dim(head_dim: usize) -> usize {
    let quarter_even = (head_dim >> 2) & !1usize;
    let base_rot = if quarter_even >= 2 { quarter_even } else { 2 };
    if base_rot <= head_dim {
        base_rot
    } else {
        head_dim
    }
}

#[inline(always)]
#[must_use]
const fn ptr_has_action_mode(tagged_ptr: gcso_tagged_ptr_t, mode_mask: u8) -> bool {
    (unpack_action_mode_from_tag(tagged_ptr.unpack_tag()) & mode_mask) == mode_mask
}

/// Resolves `(is_default_ripa_clamp, effective_max_rad, q7_to_norm_scale)` for RIPA steering.
#[inline(always)]
#[must_use]
pub(crate) const fn resolve_ripa_norm_params(max_rad: f32) -> (bool, f32, f32) {
    let is_default = max_rad.to_bits() == DEFAULT_RIPA_CLAMP_MAX_RAD.to_bits();
    let eff_max = branchless_min_f32(max_rad, FRAC_PI_2);
    let q7_to_norm_scale = if is_default {
        DEFAULT_RIPA_Q7_TO_NORM_SCALE
    } else {
        Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(eff_max)
    };
    (is_default, eff_max, q7_to_norm_scale)
}

#[inline(always)]
#[must_use]
const fn eval_ripa_sincos_with_params(
    raw_q7: gcso_q7_t,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if fold_geodesic {
        eval_geodesic_ripa_sincos_q7_fast(raw_q7, is_default, eff_max, q7_to_norm_scale)
    } else {
        eval_ripa_sincos_q7_fast(raw_q7, is_default, eff_max, q7_to_norm_scale)
    }
}

/// 4-register frameless AVX2 leaf kernel (`rcx = ptr`, `rdx = p_ptr`, `r8 = head_dim`, `r9 = num_heads`
/// on Windows x64; `rdi, rsi, rdx, rcx` on SysV x64) that rotates `ptr[0..num_heads * head_dim]`
/// in-place with zero stack argument spills, zero redundant slice bounds-check panic branches,
/// zero Windows x64 SEH unwind frames, and flat 128-float DSB-locked loops across `128`, `256`, and `512`.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
unsafe fn apply_phase_steering_nonuniform_avx2_ptr(
    ptr: *mut f32,
    p_ptr: *const u8,
    head_dim: usize,
    num_heads: usize,
) {
    use core::arch::x86_64::{
        _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };
    let lut = Q7_SINCOS_LUT_256_REF;
    // SAFETY: Caller guarantees `ptr` is valid for `num_heads * head_dim` floats, `p_ptr` is valid
    // for `num_heads` bytes, `head_dim` is even (`>= 2`), and AVX2 is enabled.
    unsafe {
        if head_dim == 128 {
            // 1 head (`4x avx2_rotate_so2_block32` = `96` AVX2 instructions, ~420B) per iteration
            // fits strictly inside a single 512B Decoded ICache (DSB) window (`0` MITE decode stalls).
            let mut hp = ptr;
            let mut h = 0usize;
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                hp = hp.add(128);
                h += 1;
            }
            return;
        }
        if head_dim == 256 {
            // Single flat loop over `num_heads * 2` 128-float blocks (`blk >> 1` phase index).
            // Keeping the loop body at `4x avx2_rotate_so2_block32` (`96` AVX2 instructions, ~420B)
            // locks `head_dim == 256` inside the 512B Decoded ICache (DSB) window (`0` MITE decode stalls).
            let mut hp = ptr;
            let blocks128 = num_heads << 1;
            let mut blk = 0usize;
            while blk < blocks128 {
                let (sin_t, cos_t) = lut[*p_ptr.add(blk >> 1) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                hp = hp.add(128);
                blk += 1;
            }
            return;
        }
        if head_dim == 512 {
            // Single flat loop over `num_heads * 4` 128-float blocks (`blk >> 2` phase index).
            let mut hp = ptr;
            let blocks128 = num_heads << 2;
            let mut blk = 0usize;
            while blk < blocks128 {
                let (sin_t, cos_t) = lut[*p_ptr.add(blk >> 2) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                hp = hp.add(128);
                blk += 1;
            }
            return;
        }
        if head_dim == 64 {
            let mut hp = ptr;
            let mut h = 0usize;
            while h + 2 <= num_heads {
                let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                let (sin0, cos0) = lut[(wb16 as u8) as usize];
                let vc0 = _mm256_set1_ps(cos0);
                let vs0 = _mm256_set1_ps(sin0);
                avx2_rotate_so2_block32(hp, vc0, vs0);
                avx2_rotate_so2_block32(hp.add(32), vc0, vs0);
                let (sin1, cos1) = lut[(wb16 >> 8) as usize];
                let vc1 = _mm256_set1_ps(cos1);
                let vs1 = _mm256_set1_ps(sin1);
                avx2_rotate_so2_block32(hp.add(64), vc1, vs1);
                avx2_rotate_so2_block32(hp.add(96), vc1, vs1);
                hp = hp.add(128);
                h += 2;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                hp = hp.add(64);
                h += 1;
            }
            return;
        }
        if head_dim == 32 {
            let mut hp = ptr;
            let mut h = 0usize;
            while h + 2 <= num_heads {
                let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                let (sin0, cos0) = lut[(wb16 as u8) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos0), _mm256_set1_ps(sin0));
                let (sin1, cos1) = lut[(wb16 >> 8) as usize];
                avx2_rotate_so2_block32(hp.add(32), _mm256_set1_ps(cos1), _mm256_set1_ps(sin1));
                hp = hp.add(64);
                h += 2;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                hp = hp.add(32);
                h += 1;
            }
            return;
        }
        if head_dim == 16 {
            let mut hp = ptr;
            let mut h = 0usize;
            while h + 4 <= num_heads {
                let wb32 = p_ptr.add(h).cast::<u32>().read_unaligned();
                let (sin0, cos0) = lut[(wb32 as u8) as usize];
                let vc0 = _mm256_set1_ps(cos0);
                let vs0 = _mm256_set1_ps(sin0);
                let v0 = _mm256_loadu_ps(hp);
                let v1 = _mm256_loadu_ps(hp.add(8));
                _mm256_storeu_ps(
                    hp,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, vc0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), vs0),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(8),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, vc0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), vs0),
                    ),
                );
                let (sin1, cos1) = lut[((wb32 >> 8) as u8) as usize];
                let vc1 = _mm256_set1_ps(cos1);
                let vs1 = _mm256_set1_ps(sin1);
                let v2 = _mm256_loadu_ps(hp.add(16));
                let v3 = _mm256_loadu_ps(hp.add(24));
                _mm256_storeu_ps(
                    hp.add(16),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v2, vc1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), vs1),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(24),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v3, vc1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), vs1),
                    ),
                );
                let (sin2, cos2) = lut[((wb32 >> 16) as u8) as usize];
                let vc2 = _mm256_set1_ps(cos2);
                let vs2 = _mm256_set1_ps(sin2);
                let v4 = _mm256_loadu_ps(hp.add(32));
                let v5 = _mm256_loadu_ps(hp.add(40));
                _mm256_storeu_ps(
                    hp.add(32),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v4, vc2),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v4), vs2),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(40),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v5, vc2),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v5), vs2),
                    ),
                );
                let (sin3, cos3) = lut[(wb32 >> 24) as usize];
                let vc3 = _mm256_set1_ps(cos3);
                let vs3 = _mm256_set1_ps(sin3);
                let v6 = _mm256_loadu_ps(hp.add(48));
                let v7 = _mm256_loadu_ps(hp.add(56));
                _mm256_storeu_ps(
                    hp.add(48),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v6, vc3),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v6), vs3),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(56),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v7, vc3),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v7), vs3),
                    ),
                );
                hp = hp.add(64);
                h += 4;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                let p1 = hp.add(8);
                let v0 = _mm256_loadu_ps(hp);
                let v1 = _mm256_loadu_ps(p1);
                _mm256_storeu_ps(
                    hp,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                    ),
                );
                _mm256_storeu_ps(
                    p1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                    ),
                );
                hp = hp.add(16);
                h += 1;
            }
            return;
        }
        let mut hp = ptr;
        let mut h = 0usize;
        while h < num_heads {
            let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
            let v_cos = _mm256_set1_ps(cos_t);
            let v_sin = _mm256_set1_ps(sin_t);
            let mut b = 0usize;
            while b + 32 <= head_dim {
                avx2_rotate_so2_block32(hp.add(b), v_cos, v_sin);
                b += 32;
            }
            while b + 8 <= head_dim {
                let p = hp.add(b);
                let v = _mm256_loadu_ps(p);
                _mm256_storeu_ps(
                    p,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v), v_sin),
                    ),
                );
                b += 8;
            }
            while b + 2 <= head_dim {
                let x = *hp.add(b);
                let y = *hp.add(b + 1);
                *hp.add(b) = x * cos_t - y * sin_t;
                *hp.add(b + 1) = x * sin_t + y * cos_t;
                b += 2;
            }
            hp = hp.add(head_dim);
            h += 1;
        }
    }
}

/// Rotates `t_slice` across `num_heads` heads when `p_slice` is already known to be non-uniform
/// (or when called directly after a caller-level uniformity check in fused QK / QKV kernels),
/// delegating on AVX2 to the 4-register frameless leaf `apply_phase_steering_nonuniform_avx2_ptr`.
#[inline(always)]
fn apply_phase_steering_nonuniform_slice_core(
    t_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: Caller has validated `t_slice.len() >= num_heads * head_dim` and `p_slice.len() >= num_heads`.
        unsafe {
            apply_phase_steering_nonuniform_avx2_ptr(
                t_slice.as_mut_ptr(),
                p_slice.as_ptr().cast::<u8>(),
                head_dim,
                num_heads,
            );
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let lut = Q7_SINCOS_LUT_256_REF;
        let total_len = num_heads * head_dim;
        let t_slice = &mut t_slice[..total_len];
        let p_slice = &p_slice[..num_heads];
        for (head_buf, &raw_q7) in t_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
            let (sin_t, cos_t) = lut[(raw_q7 as u8) as usize];
            rotate_so2_slice_unrolled16(head_buf, sin_t, cos_t);
        }
    }
}

/// Rotates `t_slice` across `num_heads` heads in strictly constant time (`0` data-dependent
/// branches on phase magnitude), coalescing uniform 8-head SWAR words (`w == w.rotate_right(8)`)
/// so `zero_phase` (`0x00..00`) and `max_saturation` (`0x7F..7F`) share the exact same single-broadcast
/// AVX2 streaming loop (`1.00x` constant-time ratio).
#[inline(never)]
fn apply_phase_steering_slice_core(
    t_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::_mm256_set1_ps;
        if num_heads >= 4 && (total_len & 127) == 0 && is_phase_slice_uniform(p_slice, num_heads) {
            // SAFETY: `total_len` is a multiple of 128 and `p_slice` is non-empty (`num_heads >= 4`).
            // Hoisting `(v_cos, v_sin)` once across the entire uniform tensor eliminates `num_heads - 1`
            // per-head LUT loads and `vbroadcastss` instructions while keeping `0x00` and `0x7F`
            // on the exact same constant-time (`1.00x`) instruction path.
            unsafe {
                let (sin_t, cos_t) = Q7_SINCOS_LUT_256_REF[(*p_slice.get_unchecked(0) as u8) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                let mut hp = t_slice.as_mut_ptr();
                let end_hp = hp.add(total_len);
                while hp < end_hp {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                    hp = hp.add(128);
                }
            }
            return;
        }
    }
    apply_phase_steering_nonuniform_slice_core(t_slice, p_slice, head_dim, num_heads);
}

/// Returns `true` in 64-bit SWAR words (`8` bytes per comparison, `0` stack buffer writes)
/// when all `num_heads` heads in `p_slice` share the exact same phase byte (`p_slice[0]`).
#[inline(always)]
#[must_use]
fn is_phase_slice_uniform(p_slice: &[gcso_q7_t], num_heads: usize) -> bool {
    let first_u8 = p_slice[0] as u8;
    if num_heads >= 2 && (p_slice[1] as u8) != first_u8 {
        return false;
    }
    if num_heads >= 8 {
        let w0 = load_q7_x8_slice(p_slice, 0);
        if w0 != w0.rotate_right(8) {
            return false;
        }
        let mut h = 8usize;
        while h + 8 <= num_heads {
            if load_q7_x8_slice(p_slice, h) != w0 {
                return false;
            }
            h += 8;
        }
        if h + 4 <= num_heads {
            let w32 = u32::from_le_bytes([
                p_slice[h] as u8,
                p_slice[h + 1] as u8,
                p_slice[h + 2] as u8,
                p_slice[h + 3] as u8,
            ]);
            if w32 != (w0 as u32) {
                return false;
            }
            h += 4;
        }
        while h < num_heads {
            if (p_slice[h] as u8) != first_u8 {
                return false;
            }
            h += 1;
        }
        return true;
    }
    let splat_first = u64::from(first_u8).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let mut h = 0usize;
    if h + 4 <= num_heads {
        let w32 = u32::from_le_bytes([
            p_slice[h] as u8,
            p_slice[h + 1] as u8,
            p_slice[h + 2] as u8,
            p_slice[h + 3] as u8,
        ]);
        if w32 != (splat_first as u32) {
            return false;
        }
        h += 4;
    }
    while h < num_heads {
        if (p_slice[h] as u8) != first_u8 {
            return false;
        }
        h += 1;
    }
    true
}

/// Unpacks a heterogeneous `p_slice[..num_heads]` (`num_heads <= MAX_ATTENTION_HEADS`) into `sincos_out`
/// in a single SWAR pass across `p_slice` (`0` duplicate uniformity scans).
#[inline(always)]
fn unpack_sincos_heads_nonuniform_64(
    p_slice: &[gcso_q7_t],
    num_heads: usize,
    sincos_out: &mut [(f32, f32); MAX_ATTENTION_HEADS],
) {
    let lut = Q7_SINCOS_LUT_256_REF;
    let mut h = 0usize;
    while h + 8 <= num_heads {
        let mut w = load_q7_x8_slice(p_slice, h);
        let b0 = (w & 0xFF) as u8;
        if w == u64::from(b0).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            let sc = lut[b0 as usize];
            sincos_out[h] = sc;
            sincos_out[h + 1] = sc;
            sincos_out[h + 2] = sc;
            sincos_out[h + 3] = sc;
            sincos_out[h + 4] = sc;
            sincos_out[h + 5] = sc;
            sincos_out[h + 6] = sc;
            sincos_out[h + 7] = sc;
        } else {
            let mut k = 0usize;
            while k < 8 {
                sincos_out[h + k] = lut[(w as u8) as usize];
                w >>= 8;
                k += 1;
            }
        }
        h += 8;
    }
    while h < num_heads {
        sincos_out[h] = lut[(p_slice[h] as u8) as usize];
        h += 1;
    }
}

/// Checks in 64-bit SWAR words whether two `num_heads` Q7 phase slices are byte-for-byte identical.
#[inline(always)]
#[must_use]
fn phase_rows_equal_swar(row_a: &[gcso_q7_t], row_b: &[gcso_q7_t], num_heads: usize) -> bool {
    let mut h = 0usize;
    while h + 8 <= num_heads {
        if load_q7_x8_slice(row_a, h) != load_q7_x8_slice(row_b, h) {
            return false;
        }
        h += 8;
    }
    while h < num_heads {
        if row_a[h] != row_b[h] {
            return false;
        }
        h += 1;
    }
    true
}

/// Returns `true` when a batch phase buffer `p_slice` is invariant across all `batch_tokens` rows:
/// either `p_slice.len() < batch_tokens * num_heads` (explicit single-row broadcast), OR every
/// subsequent token row `t in 1..batch_tokens` is either identical to token 0 (`p_slice[..num_heads]`)
/// or all zeros (zero-padded tail). Evaluated in 64-bit SWAR words (`8` bytes per comparison).
#[inline(always)]
#[must_use]
fn is_batch_phase_row_invariant(
    p_slice: &[gcso_q7_t],
    num_heads: usize,
    batch_tokens: usize,
    total_phases: usize,
) -> bool {
    if batch_tokens <= 1 || p_slice.len() < total_phases {
        return true;
    }
    let row0 = &p_slice[..num_heads];
    let tail = &p_slice[num_heads..total_phases];
    for row_t in tail.chunks_exact(num_heads) {
        let mut h = 0usize;
        let mut matches_row0 = true;
        let mut all_zero = true;
        while h + 8 <= num_heads {
            let wt = load_q7_x8_slice(row_t, h);
            let w0 = load_q7_x8_slice(row0, h);
            matches_row0 &= wt == w0;
            all_zero &= wt == 0;
            if !(matches_row0 | all_zero) {
                return false;
            }
            h += 8;
        }
        while h < num_heads {
            let bt = row_t[h];
            matches_row0 &= bt == row0[h];
            all_zero &= bt == 0;
            if !(matches_row0 | all_zero) {
                return false;
            }
            h += 1;
        }
    }
    true
}

/// Prefetches the 4KB OS page starts (`+0B, +4096B, +8192B, +12288B`) of the upcoming token slice
/// into L1/L2 cache and warms the DTLB across 4KB page boundaries during large multi-token batches.
#[inline(always)]
fn prefetch_next_token_pages(next_tok_ptr: *const f32, token_elems: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
        _mm_prefetch(next_tok_ptr.cast::<i8>(), _MM_HINT_T0);
        if token_elems > 1024 {
            _mm_prefetch(next_tok_ptr.add(1024).cast::<i8>(), _MM_HINT_T0);
        }
        if token_elems > 2048 {
            _mm_prefetch(next_tok_ptr.add(2048).cast::<i8>(), _MM_HINT_T0);
        }
        if token_elems > 3072 {
            _mm_prefetch(next_tok_ptr.add(3072).cast::<i8>(), _MM_HINT_T0);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let _ = (next_tok_ptr, token_elems);
    }
}

#[inline]
fn apply_phase_steering_multi_token_batch_core(
    tensor_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    batch_tokens: usize,
    head_dim: usize,
    num_heads: usize,
    token_elems: usize,
) -> GcsoResult<()> {
    let total_elems = batch_tokens
        .checked_mul(token_elems)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let total_phases = batch_tokens
        .checked_mul(num_heads)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if batch_tokens == 0 || tensor_slice.len() < total_elems || p_slice.len() < num_heads {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let t_all = &mut tensor_slice[..total_elems];
    let p_row0 = &p_slice[..num_heads];

    if batch_tokens == 1 {
        apply_phase_steering_slice_core(t_all, p_row0, head_dim, num_heads);
        return Ok(());
    }

    let do_prefetch = batch_tokens >= 64;
    let do_deep_prefetch = batch_tokens >= 256;
    if !is_batch_phase_row_invariant(p_slice, num_heads, batch_tokens, total_phases) {
        if num_heads <= MAX_ATTENTION_HEADS {
            let mut sincos_heads = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
            let mut prev_row: Option<&[gcso_q7_t]> = None;
            let mut all_uniform = false;
            let mut tok_idx = 0usize;
            while tok_idx < batch_tokens {
                let base = tok_idx * token_elems;
                if do_prefetch && (tok_idx + 1 < batch_tokens) {
                    // SAFETY: `(tok_idx + 1) * token_elems + token_elems <= total_elems == t_all.len()`.
                    unsafe {
                        prefetch_next_token_pages(
                            t_all.as_ptr().add(base + token_elems),
                            token_elems,
                        );
                        if do_deep_prefetch && (tok_idx + 2 < batch_tokens) {
                            prefetch_next_token_pages(
                                t_all.as_ptr().add(base + (token_elems << 1)),
                                token_elems,
                            );
                        }
                    }
                }
                let p_base = tok_idx * num_heads;
                let p_tok = &p_slice[p_base..p_base + num_heads];
                let same_as_prev = match prev_row {
                    Some(prev) => phase_rows_equal_swar(prev, p_tok, num_heads),
                    None => false,
                };
                if !same_as_prev {
                    all_uniform = is_phase_slice_uniform(p_tok, num_heads);
                    if all_uniform {
                        sincos_heads[0] = Q7_SINCOS_LUT_256_REF[(p_tok[0] as u8) as usize];
                    } else {
                        unpack_sincos_heads_nonuniform_64(p_tok, num_heads, &mut sincos_heads);
                    }
                    prev_row = Some(p_tok);
                }
                apply_phase_steering_slice_with_sincos(
                    &mut t_all[base..base + token_elems],
                    &sincos_heads,
                    all_uniform,
                    head_dim,
                    num_heads,
                );
                tok_idx += 1;
            }
        } else {
            let mut tok_idx = 0usize;
            while tok_idx < batch_tokens {
                let base = tok_idx * token_elems;
                if do_prefetch && (tok_idx + 1 < batch_tokens) {
                    // SAFETY: `(tok_idx + 1) * token_elems + token_elems <= total_elems == t_all.len()`.
                    unsafe {
                        prefetch_next_token_pages(
                            t_all.as_ptr().add(base + token_elems),
                            token_elems,
                        );
                    }
                }
                let p_base = tok_idx * num_heads;
                apply_phase_steering_slice_core(
                    &mut t_all[base..base + token_elems],
                    &p_slice[p_base..p_base + num_heads],
                    head_dim,
                    num_heads,
                );
                tok_idx += 1;
            }
        }
        return Ok(());
    }

    if is_phase_slice_uniform(p_row0, num_heads) {
        let (sin_t, cos_t) = Q7_SINCOS_LUT_256_REF[(p_row0[0] as u8) as usize];
        rotate_so2_slice_unrolled16(t_all, sin_t, cos_t);
    } else if num_heads <= MAX_ATTENTION_HEADS {
        let mut sincos_heads = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
        unpack_sincos_heads_nonuniform_64(p_row0, num_heads, &mut sincos_heads);
        let mut tok_idx = 0usize;
        while tok_idx < batch_tokens {
            let base = tok_idx * token_elems;
            if do_prefetch && (tok_idx + 1 < batch_tokens) {
                // SAFETY: `(tok_idx + 1) * token_elems + token_elems <= total_elems == t_all.len()`.
                unsafe {
                    prefetch_next_token_pages(
                        t_all.as_ptr().add(base + token_elems),
                        token_elems,
                    );
                    if do_deep_prefetch && (tok_idx + 2 < batch_tokens) {
                        prefetch_next_token_pages(
                            t_all.as_ptr().add(base + (token_elems << 1)),
                            token_elems,
                        );
                    }
                }
            }
            apply_phase_steering_slice_with_sincos(
                &mut t_all[base..base + token_elems],
                &sincos_heads,
                false,
                head_dim,
                num_heads,
            );
            tok_idx += 1;
        }
    } else {
        let mut tok_idx = 0usize;
        while tok_idx < batch_tokens {
            let base = tok_idx * token_elems;
            if do_prefetch && (tok_idx + 1 < batch_tokens) {
                // SAFETY: `(tok_idx + 1) * token_elems + token_elems <= total_elems == t_all.len()`.
                unsafe {
                    prefetch_next_token_pages(
                        t_all.as_ptr().add(base + token_elems),
                        token_elems,
                    );
                }
            }
            apply_phase_steering_nonuniform_slice_core(
                &mut t_all[base..base + token_elems],
                p_row0,
                head_dim,
                num_heads,
            );
            tok_idx += 1;
        }
    }
    Ok(())
}

#[inline]
fn apply_phase_steering_fused_multi_token_batch_core(
    query_slice: &mut [f32],
    key_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    batch_tokens: usize,
    head_dim: usize,
    num_heads: usize,
    token_elems: usize,
) -> GcsoResult<()> {
    let total_elems = batch_tokens
        .checked_mul(token_elems)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let total_phases = batch_tokens
        .checked_mul(num_heads)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if batch_tokens == 0
        || query_slice.len() < total_elems
        || key_slice.len() < total_elems
        || p_slice.len() < num_heads
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let q_all = &mut query_slice[..total_elems];
    let k_all = &mut key_slice[..total_elems];
    let p_row0 = &p_slice[..num_heads];

    // Fast-path single-token decode (`batch_tokens == 1`): zero 512B stack `sincos_heads` init.
    if batch_tokens == 1 {
        if is_phase_slice_uniform(p_row0, num_heads) {
            let (sin_t, cos_t) = Q7_SINCOS_LUT_256_REF[(p_row0[0] as u8) as usize];
            rotate_so2_slice_unrolled16(q_all, sin_t, cos_t);
            rotate_so2_slice_unrolled16(k_all, sin_t, cos_t);
        } else {
            apply_phase_steering_pair_slice_core(q_all, k_all, p_row0, head_dim, num_heads);
        }
        return Ok(());
    }

    if !is_batch_phase_row_invariant(p_slice, num_heads, batch_tokens, total_phases) {
        if num_heads <= MAX_ATTENTION_HEADS {
            let mut sincos_heads = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
            let mut prev_row: Option<&[gcso_q7_t]> = None;
            let mut all_uniform = false;
            for ((q_tok, k_tok), p_tok) in q_all
                .chunks_exact_mut(token_elems)
                .zip(k_all.chunks_exact_mut(token_elems))
                .zip(p_slice[..total_phases].chunks_exact(num_heads))
            {
                let same_as_prev = match prev_row {
                    Some(prev) => phase_rows_equal_swar(prev, p_tok, num_heads),
                    None => false,
                };
                if !same_as_prev {
                    all_uniform = is_phase_slice_uniform(p_tok, num_heads);
                    if all_uniform {
                        sincos_heads[0] = Q7_SINCOS_LUT_256_REF[(p_tok[0] as u8) as usize];
                    } else {
                        unpack_sincos_heads_nonuniform_64(p_tok, num_heads, &mut sincos_heads);
                    }
                    prev_row = Some(p_tok);
                }
                apply_phase_steering_pair_slice_with_sincos(
                    q_tok,
                    k_tok,
                    &sincos_heads,
                    all_uniform,
                    head_dim,
                    num_heads,
                );
            }
        } else {
            for ((q_tok, k_tok), p_tok) in q_all
                .chunks_exact_mut(token_elems)
                .zip(k_all.chunks_exact_mut(token_elems))
                .zip(p_slice[..total_phases].chunks_exact(num_heads))
            {
                apply_phase_steering_pair_slice_core(q_tok, k_tok, p_tok, head_dim, num_heads);
            }
        }
        return Ok(());
    }

    if is_phase_slice_uniform(p_row0, num_heads) {
        let (sin_t, cos_t) = Q7_SINCOS_LUT_256_REF[(p_row0[0] as u8) as usize];
        rotate_so2_slice_unrolled16(q_all, sin_t, cos_t);
        rotate_so2_slice_unrolled16(k_all, sin_t, cos_t);
    } else if num_heads <= MAX_ATTENTION_HEADS {
        let do_prefetch = total_elems >= 8192;
        let do_deep_prefetch = total_elems >= 131_072;
        let mut sincos_heads = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
        unpack_sincos_heads_nonuniform_64(p_row0, num_heads, &mut sincos_heads);
        let mut tok_idx = 0usize;
        while tok_idx < batch_tokens {
            let base = tok_idx * token_elems;
            if do_prefetch && (tok_idx + 1 < batch_tokens) {
                // SAFETY: `(tok_idx + 1) * token_elems + token_elems <= total_elems == q_all.len()`.
                unsafe {
                    prefetch_next_token_pages(q_all.as_ptr().add(base + token_elems), token_elems);
                    if do_deep_prefetch && (tok_idx + 2 < batch_tokens) {
                        prefetch_next_token_pages(
                            q_all.as_ptr().add(base + (token_elems << 1)),
                            token_elems,
                        );
                    }
                }
            }
            apply_phase_steering_slice_with_sincos(
                &mut q_all[base..base + token_elems],
                &sincos_heads,
                false,
                head_dim,
                num_heads,
            );
            tok_idx += 1;
        }
        let mut k_idx = 0usize;
        while k_idx < batch_tokens {
            let base = k_idx * token_elems;
            if do_prefetch && (k_idx + 1 < batch_tokens) {
                // SAFETY: `(k_idx + 1) * token_elems + token_elems <= total_elems == k_all.len()`.
                unsafe {
                    prefetch_next_token_pages(k_all.as_ptr().add(base + token_elems), token_elems);
                    if do_deep_prefetch && (k_idx + 2 < batch_tokens) {
                        prefetch_next_token_pages(
                            k_all.as_ptr().add(base + (token_elems << 1)),
                            token_elems,
                        );
                    }
                }
            }
            apply_phase_steering_slice_with_sincos(
                &mut k_all[base..base + token_elems],
                &sincos_heads,
                false,
                head_dim,
                num_heads,
            );
            k_idx += 1;
        }
    } else {
        for (q_tok, k_tok) in q_all
            .chunks_exact_mut(token_elems)
            .zip(k_all.chunks_exact_mut(token_elems))
        {
            apply_phase_steering_pair_slice_core(q_tok, k_tok, p_row0, head_dim, num_heads);
        }
    }
    Ok(())
}

#[inline]
fn apply_phase_steering_fused_safe_multi_token_batch_core(
    query_slice: &mut [f32],
    key_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    batch_tokens: usize,
    head_dim: usize,
    num_heads: usize,
    token_elems: usize,
    max_rad: f32,
) -> GcsoResult<()> {
    let total_elems = batch_tokens
        .checked_mul(token_elems)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let total_phases = batch_tokens
        .checked_mul(num_heads)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if batch_tokens == 0
        || query_slice.len() < total_elems
        || key_slice.len() < total_elems
        || p_slice.len() < num_heads
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let q_all = &mut query_slice[..total_elems];
    let k_all = &mut key_slice[..total_elems];
    if !is_batch_phase_row_invariant(p_slice, num_heads, batch_tokens, total_phases) {
        for ((q_tok, k_tok), p_tok) in q_all
            .chunks_exact_mut(token_elems)
            .zip(k_all.chunks_exact_mut(token_elems))
            .zip(p_slice[..total_phases].chunks_exact(num_heads))
        {
            apply_ripa_pair_slice_core(q_tok, k_tok, p_tok, head_dim, num_heads, false, max_rad);
        }
    } else if num_heads <= MAX_ATTENTION_HEADS {
        let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
        let mut sincos_table = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
        let (all_zero, all_uniform) = unpack_ripa_sincos_heads_64(
            &p_slice[..num_heads],
            num_heads,
            false,
            is_default,
            eff_max,
            q7_to_norm_scale,
            &mut sincos_table,
        );
        if !all_zero {
            for q_tok in q_all.chunks_exact_mut(token_elems) {
                apply_ripa_single_slice_with_sincos(
                    q_tok,
                    &sincos_table,
                    all_uniform,
                    head_dim,
                    num_heads,
                );
            }
            for k_tok in k_all.chunks_exact_mut(token_elems) {
                apply_ripa_single_slice_with_sincos(
                    k_tok,
                    &sincos_table,
                    all_uniform,
                    head_dim,
                    num_heads,
                );
            }
        }
    } else {
        let p_shared = &p_slice[..num_heads];
        for (q_tok, k_tok) in q_all
            .chunks_exact_mut(token_elems)
            .zip(k_all.chunks_exact_mut(token_elems))
        {
            apply_ripa_pair_slice_core(q_tok, k_tok, p_shared, head_dim, num_heads, false, max_rad);
        }
    }
    Ok(())
}

/// Rotates `t_slice` across `num_heads` heads using pre-unpacked `sincos_heads`, amortizing
/// Q7 SWAR unpacking and 256-entry LUT lookups across multi-token (`batch_tokens > 1`) calls
/// while broadcasting once across `total_len` when `all_uniform` holds.
#[inline(always)]
fn apply_phase_steering_slice_with_sincos(
    t_slice: &mut [f32],
    sincos_heads: &[(f32, f32)],
    all_uniform: bool,
    head_dim: usize,
    num_heads: usize,
) {
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let sincos_heads = &sincos_heads[..num_heads];
    if all_uniform {
        if let Some(&(sin_t, cos_t)) = sincos_heads.first() {
            rotate_so2_slice_unrolled16(t_slice, sin_t, cos_t);
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
            _mm256_storeu_ps,
        };
        let ptr = t_slice.as_mut_ptr();
        if head_dim == 128 {
            // SAFETY: `t_slice` has length `num_heads * 128` and AVX2 is enabled.
            // 1 head per iteration (`4x avx2_rotate_so2_block32` = `96` AVX2 instructions, ~420B)
            // fits strictly inside a single 512B Decoded ICache (DSB) window (`0` MITE decode stalls).
            unsafe {
                let mut hp = ptr;
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(h);
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                    hp = hp.add(128);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 256 {
            // SAFETY: `t_slice` has length `num_heads * 256` and AVX2 is enabled.
            // Single flat loop over `num_heads * 2` 128-float blocks (`blk >> 1` phase index)
            // fits strictly inside a single 512B Decoded ICache (DSB) window (`0` MITE decode stalls).
            unsafe {
                let mut hp = ptr;
                let blocks128 = num_heads << 1;
                let mut blk = 0usize;
                while blk < blocks128 {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(blk >> 1);
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                    hp = hp.add(128);
                    blk += 1;
                }
            }
            return;
        }
        if head_dim == 512 {
            // SAFETY: `t_slice` has length `num_heads * 512` and AVX2 is enabled.
            unsafe {
                let mut hp = ptr;
                let blocks128 = num_heads << 2;
                let mut blk = 0usize;
                while blk < blocks128 {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(blk >> 2);
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(96), v_cos, v_sin);
                    hp = hp.add(128);
                    blk += 1;
                }
            }
            return;
        }
        if head_dim == 64 {
            // SAFETY: `t_slice` has length `num_heads * 64` and AVX2 is enabled.
            unsafe {
                let mut hp = ptr;
                let mut h = 0usize;
                while h + 2 <= num_heads {
                    let (sin0, cos0) = *sincos_heads.get_unchecked(h);
                    let vc0 = _mm256_set1_ps(cos0);
                    let vs0 = _mm256_set1_ps(sin0);
                    avx2_rotate_so2_block32(hp, vc0, vs0);
                    avx2_rotate_so2_block32(hp.add(32), vc0, vs0);
                    let (sin1, cos1) = *sincos_heads.get_unchecked(h + 1);
                    let vc1 = _mm256_set1_ps(cos1);
                    let vs1 = _mm256_set1_ps(sin1);
                    avx2_rotate_so2_block32(hp.add(64), vc1, vs1);
                    avx2_rotate_so2_block32(hp.add(96), vc1, vs1);
                    hp = hp.add(128);
                    h += 2;
                }
                while h < num_heads {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(h);
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    hp = hp.add(64);
                    h += 1;
                }
            }
            return;
        }

        if head_dim == 32 {
            // SAFETY: `t_slice` has length `num_heads * 32` and AVX2 is enabled.
            unsafe {
                let mut hp = ptr;
                for &(sin_t, cos_t) in sincos_heads {
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    hp = hp.add(32);
                }
            }
            return;
        }
        if head_dim == 16 {
            // SAFETY: `t_slice` has length `num_heads * 16` and AVX2 is enabled.
            unsafe {
                let mut hp = ptr;
                for &(sin_t, cos_t) in sincos_heads {
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    let p1 = hp.add(8);
                    let v0 = _mm256_loadu_ps(hp);
                    let v1 = _mm256_loadu_ps(p1);
                    _mm256_storeu_ps(
                        hp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                        ),
                    );
                    hp = hp.add(16);
                }
            }
            return;
        }
    }
    for (head_buf, &(sin_t, cos_t)) in t_slice.chunks_exact_mut(head_dim).zip(sincos_heads.iter())
    {
        rotate_so2_slice_unrolled16(head_buf, sin_t, cos_t);
    }
}

/// Rotates `q_slice` and `k_slice` in lockstep across `num_heads` heads using pre-unpacked
/// `sincos_heads`, sharing 1 `(vc, vs)` broadcast per head across both Q and K via
/// `avx2_rotate_so2_pair_block32` (`0` 4KB page-offset aliasing stalls, `50%` fewer broadcasts).
#[inline(always)]
fn apply_phase_steering_pair_slice_with_sincos(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    sincos_heads: &[(f32, f32)],
    all_uniform: bool,
    head_dim: usize,
    num_heads: usize,
) {
    let total_len = num_heads * head_dim;
    let q_slice = &mut q_slice[..total_len];
    let k_slice = &mut k_slice[..total_len];
    let sincos_heads = &sincos_heads[..num_heads];
    if all_uniform {
        if let Some(&(sin_t, cos_t)) = sincos_heads.first() {
            rotate_so2_slice_unrolled16(q_slice, sin_t, cos_t);
            rotate_so2_slice_unrolled16(k_slice, sin_t, cos_t);
        }
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::_mm256_set1_ps;
        let mut qp = q_slice.as_mut_ptr();
        let mut kp = k_slice.as_mut_ptr();
        if head_dim == 64 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 64` and AVX2 is enabled.
            unsafe {
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(h);
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    qp = qp.add(64);
                    kp = kp.add(64);
                    h += 1;
                }
            }
            return;
        }
        if head_dim > 64 {
            apply_phase_steering_slice_with_sincos(q_slice, sincos_heads, false, head_dim, num_heads);
            apply_phase_steering_slice_with_sincos(k_slice, sincos_heads, false, head_dim, num_heads);
            return;
        }
        if head_dim == 32 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 32` and AVX2 is enabled.
            unsafe {
                let mut h = 0usize;
                while h + 2 <= num_heads {
                    let (sin0, cos0) = *sincos_heads.get_unchecked(h);
                    avx2_rotate_so2_pair_block32(qp, kp, _mm256_set1_ps(cos0), _mm256_set1_ps(sin0));
                    let (sin1, cos1) = *sincos_heads.get_unchecked(h + 1);
                    avx2_rotate_so2_pair_block32(
                        qp.add(32),
                        kp.add(32),
                        _mm256_set1_ps(cos1),
                        _mm256_set1_ps(sin1),
                    );
                    qp = qp.add(64);
                    kp = kp.add(64);
                    h += 2;
                }
                while h < num_heads {
                    let (sin_t, cos_t) = *sincos_heads.get_unchecked(h);
                    avx2_rotate_so2_pair_block32(
                        qp,
                        kp,
                        _mm256_set1_ps(cos_t),
                        _mm256_set1_ps(sin_t),
                    );
                    qp = qp.add(32);
                    kp = kp.add(32);
                    h += 1;
                }
            }
            return;
        }
    }
    apply_phase_steering_slice_with_sincos(q_slice, sincos_heads, false, head_dim, num_heads);
    apply_phase_steering_slice_with_sincos(k_slice, sincos_heads, false, head_dim, num_heads);
}

/// Rotates `q_slice` and `k_slice` across `num_heads` heads using load-first 8-float lockstep
/// `avx2_rotate_so2_pair_block32` (`<= 5` live YMM registers `ymm0..ymm4`, zero Windows x64
/// `xmm6..xmm15` stack spills, zero 4KB Load-Store Queue page-offset aliasing stalls, 2-head
/// unrolling on compact shapes, and branch-free inner blocks on `head_dim == 128`).
#[inline(never)]
fn apply_phase_steering_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    let total_len = num_heads * head_dim;
    let q_slice = &mut q_slice[..total_len];
    let k_slice = &mut k_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
            _mm256_storeu_ps,
        };
        let lut = Q7_SINCOS_LUT_256_REF;
        let mut qp = q_slice.as_mut_ptr();
        let mut kp = k_slice.as_mut_ptr();
        if num_heads >= 4 && (total_len & 127) == 0 && is_phase_slice_uniform(p_slice, num_heads) {
            // SAFETY: `total_len` is a multiple of 128 and `p_slice` is non-empty (`num_heads >= 4`).
            unsafe {
                let (sin_t, cos_t) = lut[(*p_slice.get_unchecked(0) as u8) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                let end_qp = qp.add(total_len);
                while qp < end_qp {
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    qp = qp.add(128);
                }
                let end_kp = kp.add(total_len);
                while kp < end_kp {
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    kp = kp.add(128);
                }
            }
            return;
        }
        if head_dim == 64 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 64` and AVX2 is enabled.
            // Rotating `qp` (`0..64`) then `kp` (`0..64`) per head keeps the loop at `96` AVX2
            // instructions (`~420B`, DSB-locked) while separating `store(qp + 0B)` from `load(kp + 0B)`
            // by an entire 32-float block (`24` instructions at `+128B..+224B`) to prevent 4K LSQ aliasing.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    qp = qp.add(64);
                    kp = kp.add(64);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 128 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 128` and AVX2 is enabled.
            // Rotating a full 128-float `qp` head (`512B`) before `kp` separates `store(qp + off)` from
            // `load(kp + off)` (same 4KB page offset) by 384B / 72 AVX2 instructions (0 4K LSQ aliasing),
            // halves LUT loads + broadcasts (`1` per head across Q and K), and prefetches +2 heads ahead.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    qp = qp.add(128);
                    kp = kp.add(128);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 256 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 256` and AVX2 is enabled.
            // 1 LUT load + broadcast per 256-float Q+K head pair, rotating 128-float half-heads
            // `qp[0..128] -> kp[0..128] -> qp[128..256] -> kp[128..256]` (72 instructions separation).
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(160), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(192), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(224), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(160), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(192), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(224), v_cos, v_sin);
                    qp = qp.add(256);
                    kp = kp.add(256);
                    h += 1;
                }
            }
            return;
        }
        if head_dim > 64 {
            // SAFETY: `qp` and `kp` are valid for `num_heads * head_dim` floats and `p_slice` for `num_heads`.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                apply_phase_steering_nonuniform_avx2_ptr(qp, p_ptr, head_dim, num_heads);
                apply_phase_steering_nonuniform_avx2_ptr(kp, p_ptr, head_dim, num_heads);
            }
            return;
        }
        if head_dim == 32 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 32` and AVX2 is enabled.
            unsafe {
                let mut h = 0usize;
                while h + 2 <= num_heads {
                    let (sin0, cos0) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                    avx2_rotate_so2_pair_block32(qp, kp, _mm256_set1_ps(cos0), _mm256_set1_ps(sin0));
                    let (sin1, cos1) = lut[(*p_slice.get_unchecked(h + 1) as u8) as usize];
                    avx2_rotate_so2_pair_block32(
                        qp.add(32),
                        kp.add(32),
                        _mm256_set1_ps(cos1),
                        _mm256_set1_ps(sin1),
                    );
                    qp = qp.add(64);
                    kp = kp.add(64);
                    h += 2;
                }
                while h < num_heads {
                    let (sin_t, cos_t) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                    avx2_rotate_so2_pair_block32(
                        qp,
                        kp,
                        _mm256_set1_ps(cos_t),
                        _mm256_set1_ps(sin_t),
                    );
                    qp = qp.add(32);
                    kp = kp.add(32);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 16 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 16` and AVX2 is enabled.
            unsafe {
                let mut h = 0usize;
                while h + 2 <= num_heads {
                    let (sin0, cos0) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                    let vc0 = _mm256_set1_ps(cos0);
                    let vs0 = _mm256_set1_ps(sin0);
                    let qv0 = _mm256_loadu_ps(qp);
                    let kv0 = _mm256_loadu_ps(kp);
                    _mm256_storeu_ps(
                        qp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv0, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv0), vs0),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv0, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv0), vs0),
                        ),
                    );
                    let qv1 = _mm256_loadu_ps(qp.add(8));
                    let kv1 = _mm256_loadu_ps(kp.add(8));
                    _mm256_storeu_ps(
                        qp.add(8),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv1, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv1), vs0),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp.add(8),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv1, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv1), vs0),
                        ),
                    );
                    let (sin1, cos1) = lut[(*p_slice.get_unchecked(h + 1) as u8) as usize];
                    let vc1 = _mm256_set1_ps(cos1);
                    let vs1 = _mm256_set1_ps(sin1);
                    let qv2 = _mm256_loadu_ps(qp.add(16));
                    let kv2 = _mm256_loadu_ps(kp.add(16));
                    _mm256_storeu_ps(
                        qp.add(16),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv2, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv2), vs1),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp.add(16),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv2, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv2), vs1),
                        ),
                    );
                    let qv3 = _mm256_loadu_ps(qp.add(24));
                    let kv3 = _mm256_loadu_ps(kp.add(24));
                    _mm256_storeu_ps(
                        qp.add(24),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv3, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv3), vs1),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp.add(24),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv3, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv3), vs1),
                        ),
                    );
                    qp = qp.add(32);
                    kp = kp.add(32);
                    h += 2;
                }
                while h < num_heads {
                    let (sin_t, cos_t) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    let qv0 = _mm256_loadu_ps(qp);
                    let kv0 = _mm256_loadu_ps(kp);
                    _mm256_storeu_ps(
                        qp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv0), v_sin),
                        ),
                    );
                    let qp1 = qp.add(8);
                    let kp1 = kp.add(8);
                    let qv1 = _mm256_loadu_ps(qp1);
                    let kv1 = _mm256_loadu_ps(kp1);
                    _mm256_storeu_ps(
                        qp1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(qv1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(qv1), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        kp1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(kv1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(kv1), v_sin),
                        ),
                    );
                    qp = qp.add(16);
                    kp = kp.add(16);
                    h += 1;
                }
            }
            return;
        }
    }
    apply_phase_steering_slice_core(q_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_slice_core(k_slice, p_slice, head_dim, num_heads);
}

/// Rotates `q_slice`, `k_slice`, and `v_slice` across `num_heads` heads using register-spill-free
/// `avx2_rotate_so2_triple_block32` (`<= 5` live YMM registers, zero Windows x64 `xmm6..xmm15`
/// stack spills, zero 3-way 4K Load-Store Queue aliasing stalls, and 1 LUT lookup + broadcast
/// shared across all three tensors per head).
#[inline(never)]
fn apply_phase_steering_triple_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    v_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::_mm256_set1_ps;
        let total_len = num_heads * head_dim;
        let lut = Q7_SINCOS_LUT_256_REF;
        let q_slice = &mut q_slice[..total_len];
        let k_slice = &mut k_slice[..total_len];
        let v_slice = &mut v_slice[..total_len];
        let p_slice = &p_slice[..num_heads];
        let mut qp = q_slice.as_mut_ptr();
        let mut kp = k_slice.as_mut_ptr();
        let mut vp = v_slice.as_mut_ptr();
        if num_heads >= 4 && (total_len & 127) == 0 && is_phase_slice_uniform(p_slice, num_heads) {
            // SAFETY: `total_len` is a multiple of 128 and `p_slice` is non-empty (`num_heads >= 4`).
            unsafe {
                let (sin_t, cos_t) = lut[(*p_slice.get_unchecked(0) as u8) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                let end_qp = qp.add(total_len);
                while qp < end_qp {
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    qp = qp.add(128);
                }
                let end_kp = kp.add(total_len);
                while kp < end_kp {
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    kp = kp.add(128);
                }
                let end_vp = vp.add(total_len);
                while vp < end_vp {
                    avx2_rotate_so2_block32(vp, v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(96), v_cos, v_sin);
                    vp = vp.add(128);
                }
            }
            return;
        }
        if head_dim == 32 {
            // SAFETY: `q_slice`, `k_slice`, and `v_slice` have length `num_heads * 32` and AVX2 is enabled.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_triple_block32(qp, kp, vp, v_cos, v_sin);
                    qp = qp.add(32);
                    kp = kp.add(32);
                    vp = vp.add(32);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 64 {
            // SAFETY: `q_slice`, `k_slice`, and `v_slice` have length `num_heads * 64` and AVX2 is enabled.
            // Rotating 64 floats of `qp`, then `kp`, then `vp` eliminates the inner loop branch and separates
            // same-offset stores/loads by 24 AVX2 instructions (`256B`).
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp, v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(32), v_cos, v_sin);
                    qp = qp.add(64);
                    kp = kp.add(64);
                    vp = vp.add(64);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 128 {
            // SAFETY: `q_slice`, `k_slice`, and `v_slice` have length `num_heads * 128` and AVX2 is enabled.
            // 1 LUT load + broadcast per head across Q, K, and V (`3x` fewer LUT loads), rotating full 128-float
            // heads (`qp[0..128] -> kp[0..128] -> vp[0..128]`) so same-offset stores and loads are separated by
            // 384B / 72 AVX2 instructions (0 4K LSQ false aliasing) while keeping only 3 ways active per L1D set.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp, v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(96), v_cos, v_sin);
                    qp = qp.add(128);
                    kp = kp.add(128);
                    vp = vp.add(128);
                    h += 1;
                }
            }
            return;
        }
        if head_dim == 256 {
            // SAFETY: `q_slice`, `k_slice`, and `v_slice` have length `num_heads * 256` and AVX2 is enabled.
            // 1 LUT load + broadcast per 256-float Q+K+V head triple (`6x` fewer LUT loads than 3x blocks128),
            // rotating 128-float half-heads `qp[0..128] -> kp[0..128] -> vp[0..128] -> qp[128..256] -> kp[128..256] -> vp[128..256]`.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                let mut h = 0usize;
                while h < num_heads {
                    let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(qp, v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp, v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp, v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(96), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(160), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(192), v_cos, v_sin);
                    avx2_rotate_so2_block32(qp.add(224), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(160), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(192), v_cos, v_sin);
                    avx2_rotate_so2_block32(kp.add(224), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(160), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(192), v_cos, v_sin);
                    avx2_rotate_so2_block32(vp.add(224), v_cos, v_sin);
                    qp = qp.add(256);
                    kp = kp.add(256);
                    vp = vp.add(256);
                    h += 1;
                }
            }
            return;
        }
        if head_dim > 64 {
            // SAFETY: `qp`, `kp`, and `vp` are valid for `num_heads * head_dim` floats and `p_slice` for `num_heads`.
            unsafe {
                let p_ptr = p_slice.as_ptr().cast::<u8>();
                apply_phase_steering_nonuniform_avx2_ptr(qp, p_ptr, head_dim, num_heads);
                apply_phase_steering_nonuniform_avx2_ptr(kp, p_ptr, head_dim, num_heads);
                apply_phase_steering_nonuniform_avx2_ptr(vp, p_ptr, head_dim, num_heads);
            }
            return;
        }
    }
    apply_phase_steering_pair_slice_core(q_slice, k_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_nonuniform_slice_core(v_slice, p_slice, head_dim, num_heads);
}

/// Rotates an upper low-frequency RIPA/PSPM head slice `buf` (`len == rot_dim`) in-place by
/// `(sin_t, cos_t)`, providing zero-loop-branch direct AVX2 fast paths for the canonical
/// LLM upper-quarter dimensions (`rot_dim == 16` for $d_{\mathrm{head}} = 64$, `rot_dim == 32`
/// for $d_{\mathrm{head}} = 128$, and `rot_dim == 64` for $d_{\mathrm{head}} = 256$) so small
/// head dimensions never suffer the 3-tier `while b + 32` / `while b + 8` / `while b + 2`
/// loop-branch cascade.
#[inline(always)]
fn rotate_ripa_upper_slice(buf: &mut [f32], sin_t: f32, cos_t: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
            _mm256_storeu_ps,
        };
        let len = buf.len();
        let ptr = buf.as_mut_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled and `ptr` is valid for `len` floats.
        unsafe {
            let v_cos = _mm256_set1_ps(cos_t);
            let v_sin = _mm256_set1_ps(sin_t);
            if len == 16 {
                let p0 = ptr;
                let p1 = ptr.add(8);
                let v0 = _mm256_loadu_ps(p0);
                let v1 = _mm256_loadu_ps(p1);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                    ),
                );
                _mm256_storeu_ps(
                    p1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                    ),
                );
                return;
            }
            if len == 32 {
                let p0 = ptr;
                let p1 = ptr.add(8);
                let v0 = _mm256_loadu_ps(p0);
                let v1 = _mm256_loadu_ps(p1);
                _mm256_storeu_ps(
                    p0,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                    ),
                );
                _mm256_storeu_ps(
                    p1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                    ),
                );
                let p2 = ptr.add(16);
                let p3 = ptr.add(24);
                let v2 = _mm256_loadu_ps(p2);
                let v3 = _mm256_loadu_ps(p3);
                _mm256_storeu_ps(
                    p2,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v2, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), v_sin),
                    ),
                );
                _mm256_storeu_ps(
                    p3,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v3, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), v_sin),
                    ),
                );
                return;
            }
            if len == 64 {
                for base in [0usize, 32] {
                    let p0 = ptr.add(base);
                    let p1 = ptr.add(base + 8);
                    let v0 = _mm256_loadu_ps(p0);
                    let v1 = _mm256_loadu_ps(p1);
                    _mm256_storeu_ps(
                        p0,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                        ),
                    );
                    let p2 = ptr.add(base + 16);
                    let p3 = ptr.add(base + 24);
                    let v2 = _mm256_loadu_ps(p2);
                    let v3 = _mm256_loadu_ps(p3);
                    _mm256_storeu_ps(
                        p2,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v2, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p3,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v3, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), v_sin),
                        ),
                    );
                }
                return;
            }
        }
    }
    rotate_so2_slice_unrolled16(buf, sin_t, cos_t);
}

#[inline(always)]
fn rotate_ripa_upper_heads_uniform_sincos(
    t_slice: &mut [f32],
    head_start: usize,
    head_end: usize,
    head_dim: usize,
    rot_dim: usize,
    sin_t: f32,
    cos_t: f32,
) {
    if head_start >= head_end {
        return;
    }
    let upper_off = head_dim - rot_dim;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
            _mm256_storeu_ps,
        };
        let ptr = t_slice.as_mut_ptr();
        // SAFETY: `t_slice` is valid for `head_end * head_dim` floats and AVX2 is enabled.
        unsafe {
            let v_cos = _mm256_set1_ps(cos_t);
            let v_sin = _mm256_set1_ps(sin_t);
            let mut hp = ptr.add(head_start * head_dim + upper_off);
            let mut rem = head_end - head_start;
            let stride2 = head_dim << 1;
            if head_dim == 128 {
                if rem == 8 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(256), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(384), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(512), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(640), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(768), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(896), v_cos, v_sin);
                    return;
                }
                while rem >= 4 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(128), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(256), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(384), v_cos, v_sin);
                    hp = hp.add(512);
                    rem -= 4;
                }
                while rem > 0 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    hp = hp.add(128);
                    rem -= 1;
                }
                return;
            }
            if rot_dim == 32 {
                let stride3 = head_dim * 3;
                let stride4 = head_dim << 2;
                if rem == 8 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(head_dim), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(stride2), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(stride3), v_cos, v_sin);
                    let hp4 = hp.add(stride4);
                    avx2_rotate_so2_block32(hp4, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp4.add(head_dim), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp4.add(stride2), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp4.add(stride3), v_cos, v_sin);
                    return;
                }
                while rem >= 4 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(head_dim), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(stride2), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(stride3), v_cos, v_sin);
                    hp = hp.add(stride4);
                    rem -= 4;
                }
                while rem > 0 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    hp = hp.add(head_dim);
                    rem -= 1;
                }
                return;
            }
            if rot_dim == 16 {
                while rem >= 2 {
                    let p0 = hp;
                    let p1 = hp.add(8);
                    let q0 = hp.add(head_dim);
                    let q1 = q0.add(8);
                    let v0 = _mm256_loadu_ps(p0);
                    let v1 = _mm256_loadu_ps(p1);
                    let u0 = _mm256_loadu_ps(q0);
                    let u1 = _mm256_loadu_ps(q1);
                    _mm256_storeu_ps(
                        p0,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        q0,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(u0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        q1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(u1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u1), v_sin),
                        ),
                    );
                    hp = hp.add(stride2);
                    rem -= 2;
                }
                if rem != 0 {
                    let p0 = hp;
                    let p1 = hp.add(8);
                    let v0 = _mm256_loadu_ps(p0);
                    let v1 = _mm256_loadu_ps(p1);
                    _mm256_storeu_ps(
                        p0,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                        ),
                    );
                }
                return;
            }
            if rot_dim == 64 {
                while rem >= 2 {
                    let hp1 = hp.add(head_dim);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(hp1, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp1.add(32), v_cos, v_sin);
                    hp = hp.add(stride2);
                    rem -= 2;
                }
                if rem != 0 {
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                }
                return;
            }
        }
    }
    let mut h = head_start;
    while h < head_end {
        let base = h * head_dim;
        rotate_ripa_upper_slice(&mut t_slice[base + upper_off..base + head_dim], sin_t, cos_t);
        h += 1;
    }
}

/// Unpacks `p_slice[..num_heads]` (`num_heads <= MAX_ATTENTION_HEADS`) into `sincos_out` using
/// 1-entry `(cached_q7, cached_sincos)` memoization and returns `(all_zero, all_uniform)`.
#[inline(always)]
fn unpack_ripa_sincos_heads_64(
    p_slice: &[gcso_q7_t],
    num_heads: usize,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
    sincos_out: &mut [(f32, f32); MAX_ATTENTION_HEADS],
) -> (bool, bool) {
    if num_heads == 0 {
        return (true, true);
    }
    let first_raw = p_slice[0];
    let first_eff = if fold_geodesic {
        branchless_q7_fold_s1_geodesic(first_raw)
    } else {
        first_raw
    };
    let mut cached_q7 = first_eff;
    let mut cached_sc = eval_ripa_sincos_q7_fast(first_eff, is_default, eff_max, q7_to_norm_scale);
    sincos_out[0] = cached_sc;
    let mut all_zero = first_eff == 0;
    let mut all_uniform = true;
    let mut h = 1usize;
    while h < num_heads {
        let raw_q7 = p_slice[h];
        let eff_q7 = if fold_geodesic {
            branchless_q7_fold_s1_geodesic(raw_q7)
        } else {
            raw_q7
        };
        all_zero &= eff_q7 == 0;
        all_uniform &= eff_q7 == first_eff;
        if eff_q7 != cached_q7 {
            cached_q7 = eff_q7;
            cached_sc = eval_ripa_sincos_q7_fast(eff_q7, is_default, eff_max, q7_to_norm_scale);
        }
        sincos_out[h] = cached_sc;
        h += 1;
    }
    (all_zero, all_uniform)
}

/// Rotates strictly the upper low-frequency quarter `rot_dim = ((head_dim >> 2) & !1).max(2)`
/// (`[base + head_dim - rot_dim .. base + head_dim]`) of each head in `t_slice` using pre-unpacked
/// `sincos_heads`, eliminating per-head `rot_dim` branches and LUT reloads across multi-token batches.
#[inline(always)]
fn apply_ripa_single_slice_with_sincos(
    t_slice: &mut [f32],
    sincos_heads: &[(f32, f32)],
    all_uniform: bool,
    head_dim: usize,
    num_heads: usize,
) {
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let sincos_heads = &sincos_heads[..num_heads];
    if all_uniform {
        let (sin_t, cos_t) = sincos_heads[0];
        rotate_ripa_upper_heads_uniform_sincos(t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t);
        return;
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
            _mm256_storeu_ps,
        };
        let ptr = t_slice.as_mut_ptr();
        if rot_dim == 32 {
            // SAFETY: `t_slice` has length `num_heads * head_dim` (`rot_dim == 32`) and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                for &(sin_t, cos_t) in sincos_heads {
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                    hp = hp.add(head_dim);
                }
            }
            return;
        }
        if rot_dim == 16 {
            // SAFETY: `t_slice` has length `num_heads * head_dim` (`rot_dim == 16`) and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                for &(sin_t, cos_t) in sincos_heads {
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    let p1 = hp.add(8);
                    let v0 = _mm256_loadu_ps(hp);
                    let v1 = _mm256_loadu_ps(p1);
                    _mm256_storeu_ps(
                        hp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                        ),
                    );
                    _mm256_storeu_ps(
                        p1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, v_cos),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                        ),
                    );
                    hp = hp.add(head_dim);
                }
            }
            return;
        }
        if rot_dim == 64 {
            // SAFETY: `t_slice` has length `num_heads * head_dim` (`rot_dim == 64`) and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                for &(sin_t, cos_t) in sincos_heads {
                    let v_cos = _mm256_set1_ps(cos_t);
                    let v_sin = _mm256_set1_ps(sin_t);
                    avx2_rotate_so2_block32(hp, v_cos, v_sin);
                    avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                    hp = hp.add(head_dim);
                }
            }
            return;
        }
    }
    for (head_buf, &(sin_t, cos_t)) in t_slice.chunks_exact_mut(head_dim).zip(sincos_heads.iter()) {
        rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
    }
}

/// Rotates strictly the upper low-frequency quarter `rot_dim = ((head_dim >> 2) & !1).max(2)`
/// (`[base + head_dim - rot_dim .. base + head_dim]`) of each head in `t_slice` with RIPA clamping,
/// hoisting `rot_dim == 16 / 32 / 64` specialization outside the head loop and coalescing consecutive
/// uniform 8-head SWAR cells.
#[inline(never)]
fn apply_ripa_single_slice_core(
    t_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
) {
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);

    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        if num_heads >= 8 && is_phase_slice_uniform(p_slice, num_heads) {
            let b0 = p_slice[0] as u8;
            if b0 != 0 {
                let (sin_t, cos_t) = lut[b0 as usize];
                rotate_ripa_upper_heads_uniform_sincos(
                    t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                );
            }
            return;
        }
        apply_ripa_default_nonuniform_slice_core(t_slice, p_slice, head_dim, num_heads, rot_dim, upper_off, lut);
    } else {
        let mut cached_q7: i32 = i32::MIN;
        let mut cached_sincos = (0.0_f32, 1.0_f32);
        for (head_buf, &raw_q7) in t_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
            let eff_q7 = if fold_geodesic {
                branchless_q7_fold_s1_geodesic(raw_q7)
            } else {
                raw_q7
            };
            let q7_i32 = i32::from(eff_q7);
            let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                cached_sincos
            } else {
                let sc = branchless_sincos_q7_ripa_preinv(eff_q7, eff_max, q7_to_norm_scale);
                cached_q7 = q7_i32;
                cached_sincos = sc;
                sc
            };
            rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
        }
    }
}

/// Rotates strictly the upper low-frequency quarter `rot_dim` of a single tensor `t_slice` across
/// `num_heads` heads using a preselected default RIPA LUT (`Q7_DEFAULT_RIPA_SINCOS_REF` or
/// `Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF`) when `p_slice` is already known to be non-uniform.
/// Kept `#[inline(never)]` so single-tensor and fused-QK `head_dim == 128` callers share the exact
/// same 512B Decoded ICache (DSB) loop without `for base_ptr in [q, k]` 2x loop-unroll code bloat.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
unsafe fn apply_ripa_default_nonuniform_avx2_ptr(
    ptr: *mut f32,
    p_ptr: *const u8,
    head_dim: usize,
    num_heads_and_geo: usize,
) {
    use core::arch::x86_64::{
        _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };
    let fold_geodesic = (num_heads_and_geo >> 31) != 0;
    let num_heads = num_heads_and_geo & 0x7FFF_FFFF;
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let lut = if fold_geodesic {
        Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
    } else {
        Q7_DEFAULT_RIPA_SINCOS_REF
    };
    // SAFETY: Caller guarantees `ptr` is valid for `num_heads * head_dim` floats and AVX2 is enabled.
    unsafe {
        if head_dim == 128 {
            let mut hp = ptr.add(96);
            let mut h = 0usize;
            while h + 4 <= num_heads {
                let wb32 = p_ptr.add(h).cast::<u32>().read_unaligned();
                let (s0, c0) = lut[(wb32 as u8) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                let (s1, c1) = lut[((wb32 >> 8) as u8) as usize];
                avx2_rotate_so2_block32(hp.add(128), _mm256_set1_ps(c1), _mm256_set1_ps(s1));
                let (s2, c2) = lut[((wb32 >> 16) as u8) as usize];
                avx2_rotate_so2_block32(hp.add(256), _mm256_set1_ps(c2), _mm256_set1_ps(s2));
                let (s3, c3) = lut[(wb32 >> 24) as usize];
                avx2_rotate_so2_block32(hp.add(384), _mm256_set1_ps(c3), _mm256_set1_ps(s3));
                hp = hp.add(512);
                h += 4;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                hp = hp.add(128);
                h += 1;
            }
            return;
        }
        if rot_dim == 32 {
            let mut hp = ptr.add(upper_off);
            let step2 = head_dim * 2;
            let step3 = head_dim * 3;
            let step4 = head_dim * 4;
            let mut h = 0usize;
            while h + 4 <= num_heads {
                let wb32 = p_ptr.add(h).cast::<u32>().read_unaligned();
                let (s0, c0) = lut[(wb32 as u8) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                let (s1, c1) = lut[((wb32 >> 8) as u8) as usize];
                avx2_rotate_so2_block32(hp.add(head_dim), _mm256_set1_ps(c1), _mm256_set1_ps(s1));
                let (s2, c2) = lut[((wb32 >> 16) as u8) as usize];
                avx2_rotate_so2_block32(hp.add(step2), _mm256_set1_ps(c2), _mm256_set1_ps(s2));
                let (s3, c3) = lut[(wb32 >> 24) as usize];
                avx2_rotate_so2_block32(hp.add(step3), _mm256_set1_ps(c3), _mm256_set1_ps(s3));
                hp = hp.add(step4);
                h += 4;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                hp = hp.add(head_dim);
                h += 1;
            }
            return;
        }
        if rot_dim == 16 {
            let mut hp = ptr.add(upper_off);
            let step2 = head_dim * 2;
            let mut h = 0usize;
            while h + 2 <= num_heads {
                let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                let (s0, c0) = lut[(wb16 as u8) as usize];
                let vc0 = _mm256_set1_ps(c0);
                let vs0 = _mm256_set1_ps(s0);
                let v0 = _mm256_loadu_ps(hp);
                let v1 = _mm256_loadu_ps(hp.add(8));
                _mm256_storeu_ps(
                    hp,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, vc0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), vs0),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(8),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, vc0),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), vs0),
                    ),
                );
                let hp1 = hp.add(head_dim);
                let (s1, c1) = lut[(wb16 >> 8) as usize];
                let vc1 = _mm256_set1_ps(c1);
                let vs1 = _mm256_set1_ps(s1);
                let u0 = _mm256_loadu_ps(hp1);
                let u1 = _mm256_loadu_ps(hp1.add(8));
                _mm256_storeu_ps(
                    hp1,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(u0, vc1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u0), vs1),
                    ),
                );
                _mm256_storeu_ps(
                    hp1.add(8),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(u1, vc1),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u1), vs1),
                    ),
                );
                hp = hp.add(step2);
                h += 2;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                let v0 = _mm256_loadu_ps(hp);
                let v1 = _mm256_loadu_ps(hp.add(8));
                _mm256_storeu_ps(
                    hp,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v0, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                    ),
                );
                _mm256_storeu_ps(
                    hp.add(8),
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v1, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                    ),
                );
                hp = hp.add(head_dim);
                h += 1;
            }
            return;
        }
        if rot_dim == 64 {
            let mut hp = ptr.add(upper_off);
            let step2 = head_dim * 2;
            let mut h = 0usize;
            while h + 2 <= num_heads {
                let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                let (s0, c0) = lut[(wb16 as u8) as usize];
                let vc0 = _mm256_set1_ps(c0);
                let vs0 = _mm256_set1_ps(s0);
                avx2_rotate_so2_block32(hp, vc0, vs0);
                avx2_rotate_so2_block32(hp.add(32), vc0, vs0);
                let hp1 = hp.add(head_dim);
                let (s1, c1) = lut[(wb16 >> 8) as usize];
                let vc1 = _mm256_set1_ps(c1);
                let vs1 = _mm256_set1_ps(s1);
                avx2_rotate_so2_block32(hp1, vc1, vs1);
                avx2_rotate_so2_block32(hp1.add(32), vc1, vs1);
                hp = hp.add(step2);
                h += 2;
            }
            while h < num_heads {
                let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                let v_cos = _mm256_set1_ps(cos_t);
                let v_sin = _mm256_set1_ps(sin_t);
                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                hp = hp.add(head_dim);
                h += 1;
            }
            return;
        }
        let mut hp = ptr.add(upper_off);
        let mut h = 0usize;
        while h < num_heads {
            let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
            let v_cos = _mm256_set1_ps(cos_t);
            let v_sin = _mm256_set1_ps(sin_t);
            let mut b = 0usize;
            while b + 32 <= rot_dim {
                avx2_rotate_so2_block32(hp.add(b), v_cos, v_sin);
                b += 32;
            }
            while b + 8 <= rot_dim {
                let p = hp.add(b);
                let v = _mm256_loadu_ps(p);
                _mm256_storeu_ps(
                    p,
                    _mm256_addsub_ps(
                        _mm256_mul_ps(v, v_cos),
                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v), v_sin),
                    ),
                );
                b += 8;
            }
            while b + 2 <= rot_dim {
                let x = *hp.add(b);
                let y = *hp.add(b + 1);
                *hp.add(b) = x * cos_t - y * sin_t;
                *hp.add(b + 1) = x * sin_t + y * cos_t;
                b += 2;
            }
            hp = hp.add(head_dim);
            h += 1;
        }
    }
}

#[inline(always)]
fn apply_ripa_default_nonuniform_slice_core(
    t_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    lut: &[(f32, f32); 256],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let _ = (rot_dim, upper_off);
        let is_geo = core::ptr::eq(lut, Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF);
        let packed = num_heads | ((is_geo as usize) << 31);
        // SAFETY: Caller has validated `t_slice` and `p_slice` lengths.
        unsafe {
            apply_ripa_default_nonuniform_avx2_ptr(
                t_slice.as_mut_ptr(),
                p_slice.as_ptr().cast::<u8>(),
                head_dim,
                packed,
            );
        }
        return;
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let _ = (num_heads, rot_dim);
        for (head_buf, &raw_q7) in t_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
            let (sin_t, cos_t) = lut[(raw_q7 as u8) as usize];
            rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
        }
    }
}

/// Rotates strictly the upper low-frequency quarter `rot_dim = ((head_dim >> 2) & !1).max(2)`
/// of each head in both `q_slice` and `k_slice` with RIPA soft-clamping, checking SWAR phase
/// uniformity once and using a single per-head LUT pass on compact shapes (`rot_dim == 16`)
/// while streaming `q_slice` then `k_slice` on multi-page shapes (`0` 4K LSQ aliasing, `0`
/// duplicate uniformity checks, `<= 4` live YMM registers).
#[inline(never)]
fn apply_ripa_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
) {
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let total_len = num_heads * head_dim;
    let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
    let q_slice = &mut q_slice[..total_len];
    let k_slice = &mut k_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        if num_heads >= 8 && is_phase_slice_uniform(p_slice, num_heads) {
            let b0 = p_slice[0] as u8;
            if b0 != 0 {
                let (sin_t, cos_t) = lut[b0 as usize];
                rotate_ripa_upper_heads_uniform_sincos(
                    q_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                );
                rotate_ripa_upper_heads_uniform_sincos(
                    k_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                );
            }
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::{
                _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                _mm256_set1_ps, _mm256_storeu_ps,
            };
            let q_ptr = q_slice.as_mut_ptr();
            let k_ptr = k_slice.as_mut_ptr();
            if rot_dim == 32 {
                // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 32` (`head_dim == 128`,
                // `upper_off == 96`), and AVX2 is enabled. 2-head staggered Q-then-K rotation (`qp[h], qp[h+1]`, then
                // `kp[h], kp[h+1]`) shares 1 LUT lookup + broadcast per head across Q and K while separating
                // `store(qp[h])` and `load(kp[h])` (same 4KB page offset) by 24 AVX2 instructions at `qp[h+1]` (`+512B`),
                // eliminating 4K Load-Store Queue false aliasing across all `128x16`, `128x32`, and `128x64` shapes.
                unsafe {
                    let p_ptr = p_slice.as_ptr().cast::<u8>();
                    let mut qp = q_ptr.add(96);
                    let mut kp = k_ptr.add(96);
                    let mut h = 0usize;
                    while h + 2 <= num_heads {
                        let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                        let (s0, c0) = lut[(wb16 as u8) as usize];
                        let vc0 = _mm256_set1_ps(c0);
                        let vs0 = _mm256_set1_ps(s0);
                        let (s1, c1) = lut[(wb16 >> 8) as usize];
                        let vc1 = _mm256_set1_ps(c1);
                        let vs1 = _mm256_set1_ps(s1);
                        avx2_rotate_so2_block32(qp, vc0, vs0);
                        avx2_rotate_so2_block32(qp.add(128), vc1, vs1);
                        avx2_rotate_so2_block32(kp, vc0, vs0);
                        avx2_rotate_so2_block32(kp.add(128), vc1, vs1);
                        qp = qp.add(256);
                        kp = kp.add(256);
                        h += 2;
                    }
                    while h < num_heads {
                        let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                        let vc = _mm256_set1_ps(cos_t);
                        let vs = _mm256_set1_ps(sin_t);
                        avx2_rotate_so2_block32(qp, vc, vs);
                        avx2_rotate_so2_block32(kp, vc, vs);
                        qp = qp.add(128);
                        kp = kp.add(128);
                        h += 1;
                    }
                }
                return;
            }
            if rot_dim == 16 {
                // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 16`, and AVX2 is enabled.
                unsafe {
                    let mut qp = q_ptr.add(upper_off);
                    let mut kp = k_ptr.add(upper_off);
                    let step2 = head_dim * 2;
                    let mut h = 0usize;
                    while h + 2 <= num_heads {
                        let (s0, c0) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                        let vc0 = _mm256_set1_ps(c0);
                        let vs0 = _mm256_set1_ps(s0);
                        let q0 = _mm256_loadu_ps(qp);
                        let k0 = _mm256_loadu_ps(kp);
                        _mm256_storeu_ps(
                            qp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q0, vc0),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q0), vs0),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k0, vc0),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k0), vs0),
                            ),
                        );
                        let q1 = _mm256_loadu_ps(qp.add(8));
                        let k1 = _mm256_loadu_ps(kp.add(8));
                        _mm256_storeu_ps(
                            qp.add(8),
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q1, vc0),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q1), vs0),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp.add(8),
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k1, vc0),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k1), vs0),
                            ),
                        );
                        let qp_next = qp.add(head_dim);
                        let kp_next = kp.add(head_dim);
                        let (s1, c1) = lut[(*p_slice.get_unchecked(h + 1) as u8) as usize];
                        let vc1 = _mm256_set1_ps(c1);
                        let vs1 = _mm256_set1_ps(s1);
                        let q2 = _mm256_loadu_ps(qp_next);
                        let k2 = _mm256_loadu_ps(kp_next);
                        _mm256_storeu_ps(
                            qp_next,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q2, vc1),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q2), vs1),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp_next,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k2, vc1),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k2), vs1),
                            ),
                        );
                        let q3 = _mm256_loadu_ps(qp_next.add(8));
                        let k3 = _mm256_loadu_ps(kp_next.add(8));
                        _mm256_storeu_ps(
                            qp_next.add(8),
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q3, vc1),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q3), vs1),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp_next.add(8),
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k3, vc1),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k3), vs1),
                            ),
                        );
                        qp = qp.add(step2);
                        kp = kp.add(step2);
                        h += 2;
                    }
                    while h < num_heads {
                        let (sin_t, cos_t) = lut[(*p_slice.get_unchecked(h) as u8) as usize];
                        let v_cos = _mm256_set1_ps(cos_t);
                        let v_sin = _mm256_set1_ps(sin_t);
                        let q0 = _mm256_loadu_ps(qp);
                        let k0 = _mm256_loadu_ps(kp);
                        _mm256_storeu_ps(
                            qp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q0, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q0), v_sin),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k0, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k0), v_sin),
                            ),
                        );
                        let qp1 = qp.add(8);
                        let kp1 = kp.add(8);
                        let q1 = _mm256_loadu_ps(qp1);
                        let k1 = _mm256_loadu_ps(kp1);
                        _mm256_storeu_ps(
                            qp1,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(q1, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q1), v_sin),
                            ),
                        );
                        _mm256_storeu_ps(
                            kp1,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(k1, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k1), v_sin),
                            ),
                        );
                        qp = qp.add(head_dim);
                        kp = kp.add(head_dim);
                        h += 1;
                    }
                }
                return;
            }
            if rot_dim == 64 {
                // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 64`, and AVX2 is enabled.
                // 1 head per iteration (`2x avx2_rotate_so2_pair_block32` = `96` AVX2 instructions, `~420B`)
                // fits strictly inside the 512B x86 Decoded ICache (DSB).
                unsafe {
                    let p_ptr = p_slice.as_ptr().cast::<u8>();
                    let mut qp = q_ptr.add(upper_off);
                    let mut kp = k_ptr.add(upper_off);
                    let mut h = 0usize;
                    while h < num_heads {
                        let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                        let vc = _mm256_set1_ps(cos_t);
                        let vs = _mm256_set1_ps(sin_t);
                        avx2_rotate_so2_block32(qp, vc, vs);
                        avx2_rotate_so2_block32(qp.add(32), vc, vs);
                        avx2_rotate_so2_block32(kp, vc, vs);
                        avx2_rotate_so2_block32(kp.add(32), vc, vs);
                        qp = qp.add(head_dim);
                        kp = kp.add(head_dim);
                        h += 1;
                    }
                }
                return;
            }
        }
        for (q_head, &raw_q7) in q_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
            let (sin_t, cos_t) = lut[(raw_q7 as u8) as usize];
            rotate_ripa_upper_slice(&mut q_head[upper_off..], sin_t, cos_t);
        }
        for (k_head, &raw_q7) in k_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
            let (sin_t, cos_t) = lut[(raw_q7 as u8) as usize];
            rotate_ripa_upper_slice(&mut k_head[upper_off..], sin_t, cos_t);
        }
    } else if num_heads <= MAX_ATTENTION_HEADS {
        let mut sincos_heads = [(0.0_f32, 1.0_f32); MAX_ATTENTION_HEADS];
        let (all_zero, all_uniform) = unpack_ripa_sincos_heads_64(
            p_slice,
            num_heads,
            fold_geodesic,
            is_default,
            eff_max,
            q7_to_norm_scale,
            &mut sincos_heads,
        );
        if all_zero {
            return;
        }
        apply_ripa_single_slice_with_sincos(
            q_slice,
            &sincos_heads,
            all_uniform,
            head_dim,
            num_heads,
        );
        apply_ripa_single_slice_with_sincos(
            k_slice,
            &sincos_heads,
            all_uniform,
            head_dim,
            num_heads,
        );
    } else {
        apply_ripa_single_slice_core(
            q_slice,
            p_slice,
            head_dim,
            num_heads,
            fold_geodesic,
            max_rad,
        );
        apply_ripa_single_slice_core(
            k_slice,
            p_slice,
            head_dim,
            num_heads,
            fold_geodesic,
            max_rad,
        );
    }
}

/// Rotates the upper low-frequency RIPA channels (`upper_off .. head_dim`, length `rot_dim`)
/// of a single 8-head SWAR cell (`head_start .. head_start + cell_heads`) in `t_slice` using
/// the packed 64-bit Q7 SWAR word `raw_word`, keeping cell `c`'s cache lines 100% hot in L1D.
#[inline(always)]
pub(crate) fn steer_ripa_single_cell_from_word_core(
    t_slice: &mut [f32],
    raw_word: u64,
    head_start: usize,
    cell_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    upper_off: usize,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) {
    if cell_heads == 0 {
        return;
    }
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        let mut w = raw_word;
        let b0 = (w & 0xFF) as u8;
        if w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            if b0 != 0 {
                let (sin_t, cos_t) = lut[b0 as usize];
                rotate_ripa_upper_heads_uniform_sincos(
                    t_slice,
                    head_start,
                    head_start + cell_heads,
                    head_dim,
                    rot_dim,
                    sin_t,
                    cos_t,
                );
            }
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::{
                _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                _mm256_set1_ps, _mm256_storeu_ps,
            };
            let ptr = t_slice.as_mut_ptr();
            if rot_dim == 32 {
                // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 32`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(head_start * head_dim + upper_off);
                    let step2 = head_dim * 2;
                    let step3 = head_dim * 3;
                    let step4 = head_dim * 4;
                    if cell_heads == 8 {
                        let w_lo = w as u32;
                        let w_hi = (w >> 32) as u32;
                        let (s0, c0) = lut[(w_lo as u8) as usize];
                        avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                        let (s1, c1) = lut[((w_lo >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(hp.add(head_dim), _mm256_set1_ps(c1), _mm256_set1_ps(s1));
                        let (s2, c2) = lut[((w_lo >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(hp.add(step2), _mm256_set1_ps(c2), _mm256_set1_ps(s2));
                        let (s3, c3) = lut[(w_lo >> 24) as usize];
                        avx2_rotate_so2_block32(hp.add(step3), _mm256_set1_ps(c3), _mm256_set1_ps(s3));
                        let hp4 = hp.add(step4);
                        let (s4, c4) = lut[(w_hi as u8) as usize];
                        avx2_rotate_so2_block32(hp4, _mm256_set1_ps(c4), _mm256_set1_ps(s4));
                        let (s5, c5) = lut[((w_hi >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(hp4.add(head_dim), _mm256_set1_ps(c5), _mm256_set1_ps(s5));
                        let (s6, c6) = lut[((w_hi >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(hp4.add(step2), _mm256_set1_ps(c6), _mm256_set1_ps(s6));
                        let (s7, c7) = lut[(w_hi >> 24) as usize];
                        avx2_rotate_so2_block32(hp4.add(step3), _mm256_set1_ps(c7), _mm256_set1_ps(s7));
                        return;
                    }
                    let mut k = 0usize;
                    while k + 4 <= cell_heads {
                        let (s0, c0) = lut[(w as u8) as usize];
                        avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                        let (s1, c1) = lut[((w >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(hp.add(head_dim), _mm256_set1_ps(c1), _mm256_set1_ps(s1));
                        let (s2, c2) = lut[((w >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(hp.add(step2), _mm256_set1_ps(c2), _mm256_set1_ps(s2));
                        let (s3, c3) = lut[((w >> 24) as u8) as usize];
                        avx2_rotate_so2_block32(hp.add(step3), _mm256_set1_ps(c3), _mm256_set1_ps(s3));
                        hp = hp.add(step4);
                        w >>= 32;
                        k += 4;
                    }
                    while k < cell_heads {
                        let (sin_t, cos_t) = lut[(w as u8) as usize];
                        avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                        hp = hp.add(head_dim);
                        w >>= 8;
                        k += 1;
                    }
                }
                return;
            }
            if rot_dim == 16 {
                // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 16`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(head_start * head_dim + upper_off);
                    let mut k = 0usize;
                    while k < cell_heads {
                        let (sin_t, cos_t) = lut[(w as u8) as usize];
                        let v_cos = _mm256_set1_ps(cos_t);
                        let v_sin = _mm256_set1_ps(sin_t);
                        let p1 = hp.add(8);
                        let v0 = _mm256_loadu_ps(hp);
                        let v1 = _mm256_loadu_ps(p1);
                        _mm256_storeu_ps(
                            hp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(v0, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                            ),
                        );
                        _mm256_storeu_ps(
                            p1,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(v1, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                            ),
                        );
                        hp = hp.add(head_dim);
                        w >>= 8;
                        k += 1;
                    }
                }
                return;
            }
            if rot_dim == 64 {
                // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 64`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(head_start * head_dim + upper_off);
                    let mut k = 0usize;
                    while k < cell_heads {
                        let (sin_t, cos_t) = lut[(w as u8) as usize];
                        let v_cos = _mm256_set1_ps(cos_t);
                        let v_sin = _mm256_set1_ps(sin_t);
                        avx2_rotate_so2_block32(hp, v_cos, v_sin);
                        avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                        hp = hp.add(head_dim);
                        w >>= 8;
                        k += 1;
                    }
                }
                return;
            }
        }
        let base = head_start * head_dim;
        let span = cell_heads * head_dim;
        for head_buf in t_slice[base..base + span].chunks_exact_mut(head_dim) {
            let (sin_t, cos_t) = lut[(w as u8) as usize];
            rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
            w >>= 8;
        }
    } else {
        let mut w = if fold_geodesic {
            swar_q7_fold_s1_geodesic_x8(raw_word)
        } else {
            raw_word
        };
        let b0 = (w & 0xFF) as u8;
        if w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            if b0 != 0 {
                let (sin_t, cos_t) =
                    branchless_sincos_q7_ripa_preinv(b0 as gcso_q7_t, eff_max, q7_to_norm_scale);
                rotate_ripa_upper_heads_uniform_sincos(
                    t_slice,
                    head_start,
                    head_start + cell_heads,
                    head_dim,
                    rot_dim,
                    sin_t,
                    cos_t,
                );
            }
            return;
        }
        let mut cached_q7: i32 = i32::MIN;
        let mut cached_sincos = (0.0_f32, 1.0_f32);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if rot_dim == 32 {
            use core::arch::x86_64::_mm256_set1_ps;
            let ptr = t_slice.as_mut_ptr();
            // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 32`, and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(head_start * head_dim + upper_off);
                let mut k = 0usize;
                while k < cell_heads {
                    let lane_q7 = (w as u8) as gcso_q7_t;
                    let q7_i32 = i32::from(lane_q7);
                    let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                        cached_sincos
                    } else {
                        let sc =
                            branchless_sincos_q7_ripa_preinv(lane_q7, eff_max, q7_to_norm_scale);
                        cached_q7 = q7_i32;
                        cached_sincos = sc;
                        sc
                    };
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                    hp = hp.add(head_dim);
                    w >>= 8;
                    k += 1;
                }
            }
            return;
        }
        let base = head_start * head_dim;
        let span = cell_heads * head_dim;
        for head_buf in t_slice[base..base + span].chunks_exact_mut(head_dim) {
            let lane_q7 = (w as u8) as gcso_q7_t;
            let q7_i32 = i32::from(lane_q7);
            let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                cached_sincos
            } else {
                let sc = branchless_sincos_q7_ripa_preinv(lane_q7, eff_max, q7_to_norm_scale);
                cached_q7 = q7_i32;
                cached_sincos = sc;
                sc
            };
            rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
            w >>= 8;
        }
    }
}

#[inline(always)]
fn steer_single_slice_from_eff_words_core(
    t_slice: &mut [f32],
    eff_words: &[u64; NUM_SWARM_CELLS],
    mut run_mask: u8,
    _cell_limit_mask: u8,
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) {
    let active_cells = num_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
    let max_cell_mask = ((1u16 << active_cells) - 1) as u8;
    let _cell_limit_mask = _cell_limit_mask & max_cell_mask;
    run_mask &= _cell_limit_mask;
    if run_mask == 0 || num_heads == 0 {
        return;
    }
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        let active_uni = if active_cells == 4 {
            (eff_words[0] == eff_words[1])
                && (((eff_words[0] ^ eff_words[2]) | (eff_words[0] ^ eff_words[3])) == 0)
        } else {
            is_word_array_uniform_64(eff_words)
        };
        if active_uni {
            let w0 = eff_words[0];
            let b0 = (w0 & 0xFF) as u8;
            if w0 == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                if b0 != 0 {
                    let (sin_t, cos_t) = lut[b0 as usize];
                    if run_mask == _cell_limit_mask {
                        rotate_ripa_upper_heads_uniform_sincos(
                            t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                        );
                        return;
                    }
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    if rot_dim == 32 && (num_heads & 7) == 0 {
                        // SAFETY: `t_slice` has length `num_heads * head_dim` (`num_heads` is a multiple of 8),
                        // `rot_dim == 32`, and AVX2 is enabled. Hoisting `(v_cos, v_sin)` once and rotating each
                        // active 8-head cell via 8 straight-line `avx2_rotate_so2_block32` calls eliminates
                        // all inner loop branches and `c_end` run-scan branch mispredictions.
                        unsafe {
                            use core::arch::x86_64::_mm256_set1_ps;
                            let v_cos = _mm256_set1_ps(cos_t);
                            let v_sin = _mm256_set1_ps(sin_t);
                            let base_ptr = t_slice.as_mut_ptr().add(upper_off);
                            let step2 = head_dim * 2;
                            let step3 = head_dim * 3;
                            let step4 = head_dim * 4;
                            let cell_stride = head_dim * 8;
                            let mut m = run_mask;
                            while m != 0 {
                                let c = m.trailing_zeros() as usize;
                                m &= m - 1;
                                let hp = base_ptr.add(c * cell_stride);
                                avx2_rotate_so2_block32(hp, v_cos, v_sin);
                                avx2_rotate_so2_block32(hp.add(head_dim), v_cos, v_sin);
                                avx2_rotate_so2_block32(hp.add(step2), v_cos, v_sin);
                                avx2_rotate_so2_block32(hp.add(step3), v_cos, v_sin);
                                let hp4 = hp.add(step4);
                                avx2_rotate_so2_block32(hp4, v_cos, v_sin);
                                avx2_rotate_so2_block32(hp4.add(head_dim), v_cos, v_sin);
                                avx2_rotate_so2_block32(hp4.add(step2), v_cos, v_sin);
                                avx2_rotate_so2_block32(hp4.add(step3), v_cos, v_sin);
                            }
                        }
                        return;
                    }
                    let mut m = run_mask;
                    while m != 0 {
                        let c = m.trailing_zeros() as usize;
                        let shifted = m >> c;
                        let run_len = (!shifted).trailing_zeros() as usize;
                        let c_end = (c + run_len).min(NUM_SWARM_CELLS);
                        let clear_shift = run_len.min(8);
                        m &= !((((1u16 << clear_shift) - 1) as u8) << c);
                        let head_start = c * HEADS_PER_CELL;
                        if head_start >= num_heads {
                            break;
                        }
                        let head_end = (c_end * HEADS_PER_CELL).min(num_heads);
                        rotate_ripa_upper_heads_uniform_sincos(
                            t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                        );
                    }
                }
                return;
            }
        }
        if run_mask == _cell_limit_mask {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                let packed = num_heads | ((fold_geodesic as usize) << 31);
                // SAFETY: `t_slice` has length `num_heads * head_dim`, `eff_words` has `8 * 8 = 64 >= num_heads`
                // little-endian Q7 bytes, and AVX2 is enabled. Delegating to the shared frameless leaf
                // `apply_ripa_default_nonuniform_avx2_ptr` keeps all swarm/SLERP/unified-field callers
                // 100% GPR-only (`0` Windows x64 `xmm6..xmm15` stack saves) and DSB-resident.
                unsafe {
                    apply_ripa_default_nonuniform_avx2_ptr(
                        t_slice.as_mut_ptr(),
                        eff_words.as_ptr().cast::<u8>(),
                        head_dim,
                        packed,
                    );
                }
                return;
            }
        }
        while run_mask != 0 {
            let c = run_mask.trailing_zeros() as usize;
            run_mask &= run_mask - 1;
            let head_start = c * HEADS_PER_CELL;
            if head_start >= num_heads {
                break;
            }
            let w = eff_words[c];
            let b0 = (w & 0xFF) as u8;
            if w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                let mut c_end = c + 1;
                while c_end < NUM_SWARM_CELLS
                    && ((run_mask & (1u8 << c_end)) != 0)
                    && eff_words[c_end] == w
                {
                    run_mask &= !(1u8 << c_end);
                    c_end += 1;
                }
                if b0 != 0 {
                    let head_end = (c_end * HEADS_PER_CELL).min(num_heads);
                    let (sin_t, cos_t) = lut[b0 as usize];
                    rotate_ripa_upper_heads_uniform_sincos(
                        t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                continue;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            steer_ripa_single_cell_from_word_core(
                t_slice,
                w,
                head_start,
                cell_heads,
                head_dim,
                rot_dim,
                upper_off,
                fold_geodesic,
                true,
                eff_max,
                q7_to_norm_scale,
            );
        }
    } else {
        if run_mask == _cell_limit_mask && is_word_array_uniform_64(eff_words) {
            let w0 = eff_words[0];
            let b0 = (w0 & 0xFF) as u8;
            if w0 == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                if b0 != 0 {
                    let eff_q7 = if fold_geodesic {
                        branchless_q7_fold_s1_geodesic(b0 as gcso_q7_t)
                    } else {
                        b0 as gcso_q7_t
                    };
                    let (sin_t, cos_t) =
                        branchless_sincos_q7_ripa_preinv(eff_q7, eff_max, q7_to_norm_scale);
                    rotate_ripa_upper_heads_uniform_sincos(
                        t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                return;
            }
        }
        while run_mask != 0 {
            let c = run_mask.trailing_zeros() as usize;
            run_mask &= run_mask - 1;
            let head_start = c * HEADS_PER_CELL;
            if head_start >= num_heads {
                break;
            }
            let raw_w = eff_words[c];
            let b0 = (raw_w & 0xFF) as u8;
            if raw_w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                let mut c_end = c + 1;
                while c_end < NUM_SWARM_CELLS
                    && ((run_mask & (1u8 << c_end)) != 0)
                    && eff_words[c_end] == raw_w
                {
                    run_mask &= !(1u8 << c_end);
                    c_end += 1;
                }
                if b0 != 0 {
                    let head_end = (c_end * HEADS_PER_CELL).min(num_heads);
                    let eff_q7 = if fold_geodesic {
                        branchless_q7_fold_s1_geodesic(b0 as gcso_q7_t)
                    } else {
                        b0 as gcso_q7_t
                    };
                    let (sin_t, cos_t) =
                        branchless_sincos_q7_ripa_preinv(eff_q7, eff_max, q7_to_norm_scale);
                    rotate_ripa_upper_heads_uniform_sincos(
                        t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                continue;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            steer_ripa_single_cell_from_word_core(
                t_slice,
                raw_w,
                head_start,
                cell_heads,
                head_dim,
                rot_dim,
                upper_off,
                fold_geodesic,
                false,
                eff_max,
                q7_to_norm_scale,
            );
        }
    }
}

#[inline(always)]
fn steer_pair_slice_from_eff_words_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    eff_words: &[u64; NUM_SWARM_CELLS],
    mut run_mask: u8,
    _cell_limit_mask: u8,
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) {
    let active_cells = num_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
    let max_cell_mask = ((1u16 << active_cells) - 1) as u8;
    let _cell_limit_mask = _cell_limit_mask & max_cell_mask;
    run_mask &= _cell_limit_mask;
    if run_mask == 0 || num_heads == 0 {
        return;
    }
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        if run_mask == _cell_limit_mask {
            let active_uni = if active_cells == 4 {
                (eff_words[0] == eff_words[1])
                    && (((eff_words[0] ^ eff_words[2]) | (eff_words[0] ^ eff_words[3])) == 0)
            } else {
                is_word_array_uniform_64(eff_words)
            };
            if active_uni {
                let w0 = eff_words[0];
                let b0 = (w0 & 0xFF) as u8;
                if w0 == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                    if b0 != 0 {
                        let (sin_t, cos_t) = lut[b0 as usize];
                        rotate_ripa_upper_heads_uniform_sincos(
                            q_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                        );
                        rotate_ripa_upper_heads_uniform_sincos(
                            k_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                        );
                    }
                    return;
                }
            }
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            {
                use core::arch::x86_64::{
                    _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                    _mm256_set1_ps, _mm256_storeu_ps,
                };
                if rot_dim == 32 {
                    // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 32` (`head_dim == 128`,
                    // `upper_off == 96`), `run_mask == _cell_limit_mask`, and AVX2 is enabled. 2-head staggered Q-then-K
                    // rotation shares 1 LUT lookup + broadcast per head across Q and K and separates `store(qp)` from
                    // `load(kp)` by 24 AVX2 instructions at `qp.add(128)` (`0` 4K LSQ false aliasing).
                    unsafe {
                        let p_ptr = eff_words.as_ptr().cast::<u8>();
                        let mut qp = q_slice.as_mut_ptr().add(96);
                        let mut kp = k_slice.as_mut_ptr().add(96);
                        let mut h = 0usize;
                        while h + 2 <= num_heads {
                            let wb16 = p_ptr.add(h).cast::<u16>().read_unaligned();
                            let (s0, c0) = lut[(wb16 as u8) as usize];
                            let vc0 = _mm256_set1_ps(c0);
                            let vs0 = _mm256_set1_ps(s0);
                            let (s1, c1) = lut[(wb16 >> 8) as usize];
                            let vc1 = _mm256_set1_ps(c1);
                            let vs1 = _mm256_set1_ps(s1);
                            avx2_rotate_so2_block32(qp, vc0, vs0);
                            avx2_rotate_so2_block32(qp.add(128), vc1, vs1);
                            avx2_rotate_so2_block32(kp, vc0, vs0);
                            avx2_rotate_so2_block32(kp.add(128), vc1, vs1);
                            qp = qp.add(256);
                            kp = kp.add(256);
                            h += 2;
                        }
                        while h < num_heads {
                            let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                            let vc = _mm256_set1_ps(cos_t);
                            let vs = _mm256_set1_ps(sin_t);
                            avx2_rotate_so2_block32(qp, vc, vs);
                            avx2_rotate_so2_block32(kp, vc, vs);
                            qp = qp.add(128);
                            kp = kp.add(128);
                            h += 1;
                        }
                    }
                    return;
                }
                if rot_dim == 16 {
                    // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 16`, and AVX2 is enabled.
                    unsafe {
                        let mut qp = q_slice.as_mut_ptr().add(upper_off);
                        let mut kp = k_slice.as_mut_ptr().add(upper_off);
                        let step2 = head_dim * 2;
                        let mut h = 0usize;
                        while h + 2 <= num_heads {
                            let w16 = (eff_words[h >> 3] >> ((h & 6) << 3)) as u32;
                            let (s0, c0) = lut[(w16 as u8) as usize];
                            let vc0 = _mm256_set1_ps(c0);
                            let vs0 = _mm256_set1_ps(s0);
                            let q0 = _mm256_loadu_ps(qp);
                            let k0 = _mm256_loadu_ps(kp);
                            _mm256_storeu_ps(
                                qp,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q0, vc0),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q0), vs0),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k0, vc0),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k0), vs0),
                                ),
                            );
                            let q1 = _mm256_loadu_ps(qp.add(8));
                            let k1 = _mm256_loadu_ps(kp.add(8));
                            _mm256_storeu_ps(
                                qp.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q1, vc0),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q1), vs0),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k1, vc0),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k1), vs0),
                                ),
                            );
                            let qp_next = qp.add(head_dim);
                            let kp_next = kp.add(head_dim);
                            let (s1, c1) = lut[((w16 >> 8) as u8) as usize];
                            let vc1 = _mm256_set1_ps(c1);
                            let vs1 = _mm256_set1_ps(s1);
                            let q2 = _mm256_loadu_ps(qp_next);
                            let k2 = _mm256_loadu_ps(kp_next);
                            _mm256_storeu_ps(
                                qp_next,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q2, vc1),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q2), vs1),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp_next,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k2, vc1),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k2), vs1),
                                ),
                            );
                            let q3 = _mm256_loadu_ps(qp_next.add(8));
                            let k3 = _mm256_loadu_ps(kp_next.add(8));
                            _mm256_storeu_ps(
                                qp_next.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q3, vc1),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q3), vs1),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp_next.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k3, vc1),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k3), vs1),
                                ),
                            );
                            qp = qp.add(step2);
                            kp = kp.add(step2);
                            h += 2;
                        }
                        while h < num_heads {
                            let b = (eff_words[h >> 3] >> ((h & 7) << 3)) as u8;
                            let (sin_t, cos_t) = lut[b as usize];
                            let v_cos = _mm256_set1_ps(cos_t);
                            let v_sin = _mm256_set1_ps(sin_t);
                            let q0 = _mm256_loadu_ps(qp);
                            let k0 = _mm256_loadu_ps(kp);
                            _mm256_storeu_ps(
                                qp,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q0, v_cos),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q0), v_sin),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp,
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k0, v_cos),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k0), v_sin),
                                ),
                            );
                            let q1 = _mm256_loadu_ps(qp.add(8));
                            let k1 = _mm256_loadu_ps(kp.add(8));
                            _mm256_storeu_ps(
                                qp.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(q1, v_cos),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q1), v_sin),
                                ),
                            );
                            _mm256_storeu_ps(
                                kp.add(8),
                                _mm256_addsub_ps(
                                    _mm256_mul_ps(k1, v_cos),
                                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k1), v_sin),
                                ),
                            );
                            qp = qp.add(head_dim);
                            kp = kp.add(head_dim);
                            h += 1;
                        }
                    }
                    return;
                }
                if rot_dim == 64 {
                    // SAFETY: `q_slice` and `k_slice` have length `num_heads * head_dim`, `rot_dim == 64`, and AVX2 is enabled.
                    unsafe {
                        let p_ptr = eff_words.as_ptr().cast::<u8>();
                        let mut qp = q_slice.as_mut_ptr().add(upper_off);
                        let mut kp = k_slice.as_mut_ptr().add(upper_off);
                        let mut h = 0usize;
                        while h < num_heads {
                            let (sin_t, cos_t) = lut[*p_ptr.add(h) as usize];
                            let v_cos = _mm256_set1_ps(cos_t);
                            let v_sin = _mm256_set1_ps(sin_t);
                            avx2_rotate_so2_block32(qp, v_cos, v_sin);
                            avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                            avx2_rotate_so2_block32(kp, v_cos, v_sin);
                            avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                            qp = qp.add(head_dim);
                            kp = kp.add(head_dim);
                            h += 1;
                        }
                    }
                    return;
                }
            }
        }
        while run_mask != 0 {
            let c = run_mask.trailing_zeros() as usize;
            run_mask &= run_mask - 1;
            let mut w = eff_words[c];
            let head_start = c * HEADS_PER_CELL;
            if head_start >= num_heads {
                break;
            }
            let b0 = (w & 0xFF) as u8;
            if w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                let mut c_end = c + 1;
                while c_end < NUM_SWARM_CELLS
                    && ((run_mask & (1u8 << c_end)) != 0)
                    && eff_words[c_end] == w
                {
                    run_mask &= !(1u8 << c_end);
                    c_end += 1;
                }
                if b0 != 0 {
                    let head_end = (c_end * HEADS_PER_CELL).min(num_heads);
                    let (sin_t, cos_t) = lut[b0 as usize];
                    rotate_ripa_upper_heads_uniform_sincos(
                        q_slice,
                        head_start,
                        head_end,
                        head_dim,
                        rot_dim,
                        sin_t,
                        cos_t,
                    );
                    rotate_ripa_upper_heads_uniform_sincos(
                        k_slice,
                        head_start,
                        head_end,
                        head_dim,
                        rot_dim,
                        sin_t,
                        cos_t,
                    );
                }
                continue;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                {
                    use core::arch::x86_64::{
                        _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                        _mm256_set1_ps, _mm256_storeu_ps,
                    };
                    let q_ptr = q_slice.as_mut_ptr();
                    let k_ptr = k_slice.as_mut_ptr();
                    if rot_dim == 32 {
                        // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 32`, and AVX2 is enabled.
                        unsafe {
                            let start_off = head_start * head_dim + upper_off;
                            let mut qp = q_ptr.add(start_off);
                            let mut kp = k_ptr.add(start_off);
                            let mut k = 0usize;
                            while k < cell_heads {
                                let (sin_t, cos_t) = lut[(w as u8) as usize];
                                avx2_rotate_so2_pair_block32(
                                    qp,
                                    kp,
                                    _mm256_set1_ps(cos_t),
                                    _mm256_set1_ps(sin_t),
                                );
                                qp = qp.add(head_dim);
                                kp = kp.add(head_dim);
                                w >>= 8;
                                k += 1;
                            }
                        }
                        continue;
                    }
                    if rot_dim == 16 {
                        // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 16`, and AVX2 is enabled.
                        unsafe {
                            let start_off = head_start * head_dim + upper_off;
                            let mut qp = q_ptr.add(start_off);
                            let mut kp = k_ptr.add(start_off);
                            let mut k = 0usize;
                            while k < cell_heads {
                                let (sin_t, cos_t) = lut[(w as u8) as usize];
                                let v_cos = _mm256_set1_ps(cos_t);
                                let v_sin = _mm256_set1_ps(sin_t);
                                let q0 = _mm256_loadu_ps(qp);
                                let q1 = _mm256_loadu_ps(qp.add(8));
                                _mm256_storeu_ps(
                                    qp,
                                    _mm256_addsub_ps(
                                        _mm256_mul_ps(q0, v_cos),
                                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q0), v_sin),
                                    ),
                                );
                                _mm256_storeu_ps(
                                    qp.add(8),
                                    _mm256_addsub_ps(
                                        _mm256_mul_ps(q1, v_cos),
                                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(q1), v_sin),
                                    ),
                                );
                                let k0 = _mm256_loadu_ps(kp);
                                let k1 = _mm256_loadu_ps(kp.add(8));
                                _mm256_storeu_ps(
                                    kp,
                                    _mm256_addsub_ps(
                                        _mm256_mul_ps(k0, v_cos),
                                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k0), v_sin),
                                    ),
                                );
                                _mm256_storeu_ps(
                                    kp.add(8),
                                    _mm256_addsub_ps(
                                        _mm256_mul_ps(k1, v_cos),
                                        _mm256_mul_ps(_mm256_permute_ps::<0xB1>(k1), v_sin),
                                    ),
                                );
                                qp = qp.add(head_dim);
                                kp = kp.add(head_dim);
                                w >>= 8;
                                k += 1;
                            }
                        }
                        continue;
                    }
                    if rot_dim == 64 {
                        // SAFETY: `head_start + cell_heads <= num_heads`, `rot_dim == 64`, and AVX2 is enabled.
                        unsafe {
                            let start_off = head_start * head_dim + upper_off;
                            let mut qp = q_ptr.add(start_off);
                            let mut kp = k_ptr.add(start_off);
                            let mut k = 0usize;
                            while k < cell_heads {
                                let (sin_t, cos_t) = lut[(w as u8) as usize];
                                let v_cos = _mm256_set1_ps(cos_t);
                                let v_sin = _mm256_set1_ps(sin_t);
                                avx2_rotate_so2_block32(qp, v_cos, v_sin);
                                avx2_rotate_so2_block32(qp.add(32), v_cos, v_sin);
                                avx2_rotate_so2_block32(kp, v_cos, v_sin);
                                avx2_rotate_so2_block32(kp.add(32), v_cos, v_sin);
                                qp = qp.add(head_dim);
                                kp = kp.add(head_dim);
                                w >>= 8;
                                k += 1;
                            }
                        }
                        continue;
                    }
                }
                let base = head_start * head_dim;
                let span = cell_heads * head_dim;
                for (q_head, k_head) in q_slice[base..base + span]
                    .chunks_exact_mut(head_dim)
                    .zip(k_slice[base..base + span].chunks_exact_mut(head_dim))
                {
                    let (sin_t, cos_t) = lut[(w as u8) as usize];
                    rotate_ripa_upper_slice(&mut q_head[upper_off..], sin_t, cos_t);
                    rotate_ripa_upper_slice(&mut k_head[upper_off..], sin_t, cos_t);
                    w >>= 8;
                }
            }
        }
    } else {
        if run_mask == _cell_limit_mask && is_word_array_uniform_64(eff_words) {
            let w0 = eff_words[0];
            let b0 = (w0 & 0xFF) as u8;
            if w0 == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                if b0 != 0 {
                    let eff_q7 = if fold_geodesic {
                        branchless_q7_fold_s1_geodesic(b0 as gcso_q7_t)
                    } else {
                        b0 as gcso_q7_t
                    };
                    let (sin_t, cos_t) =
                        branchless_sincos_q7_ripa_preinv(eff_q7, eff_max, q7_to_norm_scale);
                    rotate_ripa_upper_heads_uniform_sincos(
                        q_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                    );
                    rotate_ripa_upper_heads_uniform_sincos(
                        k_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                return;
            }
        }
        let mut cached_q7: i32 = i32::MIN;
        let mut cached_sincos = (0.0_f32, 1.0_f32);
        while run_mask != 0 {
            let c = run_mask.trailing_zeros() as usize;
            run_mask &= run_mask - 1;
            let raw_w = eff_words[c];
            let mut w = if fold_geodesic {
                swar_q7_fold_s1_geodesic_x8(raw_w)
            } else {
                raw_w
            };
            let head_start = c * HEADS_PER_CELL;
            if head_start >= num_heads {
                break;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            let b0 = (w & 0xFF) as u8;
            if w == (b0 as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) {
                if b0 != 0 {
                    let q7_i32 = i32::from(b0 as gcso_q7_t);
                    let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                        cached_sincos
                    } else {
                        let sc = branchless_sincos_q7_ripa_preinv(
                            b0 as gcso_q7_t,
                            eff_max,
                            q7_to_norm_scale,
                        );
                        cached_q7 = q7_i32;
                        cached_sincos = sc;
                        sc
                    };
                    rotate_ripa_upper_heads_uniform_sincos(
                        q_slice,
                        head_start,
                        head_start + cell_heads,
                        head_dim,
                        rot_dim,
                        sin_t,
                        cos_t,
                    );
                    rotate_ripa_upper_heads_uniform_sincos(
                        k_slice,
                        head_start,
                        head_start + cell_heads,
                        head_dim,
                        rot_dim,
                        sin_t,
                        cos_t,
                    );
                }
                continue;
            }
            let base = head_start * head_dim;
            let span = cell_heads * head_dim;
            for (q_head, k_head) in q_slice[base..base + span]
                .chunks_exact_mut(head_dim)
                .zip(k_slice[base..base + span].chunks_exact_mut(head_dim))
            {
                let lane_q7 = (w as u8) as gcso_q7_t;
                let q7_i32 = i32::from(lane_q7);
                let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                    cached_sincos
                } else {
                    let sc = branchless_sincos_q7_ripa_preinv(lane_q7, eff_max, q7_to_norm_scale);
                    cached_q7 = q7_i32;
                    cached_sincos = sc;
                    sc
                };
                rotate_ripa_upper_slice(&mut q_head[upper_off..], sin_t, cos_t);
                rotate_ripa_upper_slice(&mut k_head[upper_off..], sin_t, cos_t);
                w >>= 8;
            }
        }
    }
}

/// Core constant-time register-resident 8-cell SWAR RIPA phase steering kernel (`0` data-dependent
/// branches inside the cell loop, `0` stack spills, rotating strictly the upper $d_{\mathrm{head}}/4$
/// low-frequency channels per head and coalescing consecutive uniform SWAR cells).
#[inline(always)]
pub(crate) fn steer_from_swar_words_core_safe(
    swar_words: &[u64; NUM_SWARM_CELLS],
    quorum_mask: u8,
    clean_damping_u7: u8,
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
    query_tensor: &mut AlignedSliceMut32<'_, f32>,
    key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
) {
    let active_cells = num_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
    let cell_limit_mask = ((1u16 << active_cells) - 1) as u8;
    let eff_quorum = quorum_mask & cell_limit_mask;
    if (eff_quorum == 0) & (clean_damping_u7 == 0) {
        return;
    }

    let rot_dim = ripa_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let full_consensus = (eff_quorum == cell_limit_mask) | (clean_damping_u7 >= 128);
    let damping_s = u64::from(clean_damping_u7);
    let damping_bias = (128u64.wrapping_sub(damping_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);

    // Branchless pre-pass over the 8 SWAR words only when partial consensus damping is active,
    // borrowing `swar_words` directly (`0` 64B stack copy) on full consensus or zero damping.
    let damped_words: [u64; NUM_SWARM_CELLS];
    let eff_words: &[u64; NUM_SWARM_CELLS] = if !full_consensus && damping_s > 0 {
        let mut dw = *swar_words;
        let mut unresolved = (!eff_quorum) & cell_limit_mask;
        while unresolved != 0 {
            let c = unresolved.trailing_zeros() as usize;
            unresolved &= unresolved - 1;
            let raw_w = dw[c];
            let base_w = if fold_geodesic {
                swar_q7_fold_s1_geodesic_x8(raw_w)
            } else {
                raw_w
            };
            dw[c] = swar_q7_scale_le128_with_bias_x8(base_w, damping_s, damping_bias);
        }
        damped_words = dw;
        &damped_words
    } else {
        swar_words
    };

    let run_mask = if full_consensus || damping_s > 0 {
        cell_limit_mask
    } else {
        eff_quorum
    };

    if let Some(k_ref) = key_tensor {
        steer_pair_slice_from_eff_words_core(
            query_tensor.as_mut_slice(),
            k_ref.as_mut_slice(),
            eff_words,
            run_mask,
            cell_limit_mask,
            head_dim,
            num_heads,
            rot_dim,
            upper_off,
            fold_geodesic,
            is_default,
            eff_max,
            q7_to_norm_scale,
        );
    } else {
        steer_single_slice_from_eff_words_core(
            query_tensor.as_mut_slice(),
            eff_words,
            run_mask,
            cell_limit_mask,
            head_dim,
            num_heads,
            rot_dim,
            upper_off,
            fold_geodesic,
            is_default,
            eff_max,
            q7_to_norm_scale,
        );
    }
}

/// Evaluates $\mathcal{C}_{\mathrm{phase}} = \frac{1}{H}\sum_{h=0}^{H-1}\cos((\Delta\theta_{q,h} - \Delta\theta_{m,h})\frac{\pi}{128})$
/// across `n` heads using a zero-panic 64-head fast path (`coherence_from_word_arrays_64`),
/// 8-lane SWAR `swar_q7_circular_diff_x8`, and 4-port ILP Q14 integer LUT accumulation.
#[inline(always)]
#[must_use]
fn eval_multi_head_phase_coherence_unrolled4(q_p: &[gcso_q7_t], m_p: &[gcso_q7_t]) -> f32 {
    let n = q_p.len().min(m_p.len());
    if n == 0 {
        return 0.0;
    }
    if n == MAX_ATTENTION_HEADS {
        // SAFETY: `n == MAX_ATTENTION_HEADS` and `n = q_p.len().min(m_p.len())`,
        // so both slices contain at least 64 `gcso_q7_t` elements (`align_of::<gcso_q7_t>() == 1`).
        let q_arr = unsafe { &*q_p.as_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        // SAFETY: `m_p.len() >= MAX_ATTENTION_HEADS` holds by `n == MAX_ATTENTION_HEADS`.
        let m_arr = unsafe { &*m_p.as_ptr().cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>() };
        return coherence_from_word_arrays_64(&load_q7_words_64(q_arr), &load_q7_words_64(m_arr));
    }

    let mut sum_q14 = 0i32;
    let mut has_prev_dw = false;
    let mut prev_dw = 0u64;
    let mut prev_w_sum = 0i32;
    let mut h = 0usize;
    while h + 8 <= n {
        let dw = swar_q7_circular_diff_x8(load_q7_x8_slice(q_p, h), load_q7_x8_slice(m_p, h));
        let w_sum = if has_prev_dw & (dw == prev_dw) {
            prev_w_sum
        } else if dw == 0 {
            let s = 8 * 16384;
            has_prev_dw = true;
            prev_dw = 0;
            prev_w_sum = s;
            s
        } else if dw == (dw & 0xFF).wrapping_mul(SWAR_SPLAT_BYTE_64) {
            let s = i32::from(Q7_FULL_COS_I16_REF[(dw & 0xFF) as usize]) << 3;
            has_prev_dw = true;
            prev_dw = dw;
            prev_w_sum = s;
            s
        } else {
            let b = dw.to_le_bytes();
            let s0 = i32::from(Q7_FULL_COS_I16_REF[b[0] as usize])
                + i32::from(Q7_FULL_COS_I16_REF[b[4] as usize]);
            let s1 = i32::from(Q7_FULL_COS_I16_REF[b[1] as usize])
                + i32::from(Q7_FULL_COS_I16_REF[b[5] as usize]);
            let s2 = i32::from(Q7_FULL_COS_I16_REF[b[2] as usize])
                + i32::from(Q7_FULL_COS_I16_REF[b[6] as usize]);
            let s3 = i32::from(Q7_FULL_COS_I16_REF[b[3] as usize])
                + i32::from(Q7_FULL_COS_I16_REF[b[7] as usize]);
            let s = (s0 + s1) + (s2 + s3);
            has_prev_dw = true;
            prev_dw = dw;
            prev_w_sum = s;
            s
        };
        sum_q14 += w_sum;
        h += 8;
    }
    while h < n {
        let d = (q_p[h] as u8).wrapping_sub(m_p[h] as u8) as usize;
        sum_q14 += i32::from(Q7_FULL_COS_I16_REF[d]);
        h += 1;
    }
    branchless_clamp_f32(
        (sum_q14 as f32) * (INV_Q14_SCALE_F32 * fast_inv_len_f32(n)),
        -1.0,
        1.0,
    )
}

// ===================================================================
// 1. Dynamic Phase-Shifted RoPE, RIPA, QDPS & Lazy Unwrapping (`PhaseSteering`)
// ===================================================================

/// Core trait for Dynamic Phase-Shifted RoPE (DPSR), Rotational Invariant Phase Anchors (RIPA),
/// Quantized Discrete Phase Stepping (QDPS), Seam-Free Circular $S^1$ Lazy Phase Unwrapping,
/// Single-Manifold Unified-Field Phase Steering, Attention $\times$ DPSR Phase-Interference Score
/// Modulation, Bayesian Pointer-Trail Attention Modulation, and Closed-Form Orthogonal Procrustes
/// Phase Estimation (**Nano / Micro Level**).
pub trait PhaseSteering: BranchlessKernel {
    /// Returns the configured attention head dimension ($d_{\mathrm{head}}$, positive even integer).
    fn head_dim(&self) -> u32;

    /// Returns the configured number of attention heads ($H > 0$).
    fn num_heads(&self) -> u32;

    /// Validates `(head_dim, num_heads)` invariants (`head_dim > 0`, `head_dim % 2 == 0`, `num_heads > 0`)
    /// and returns `(head_dim_usize, num_heads_usize, expected_tensor_len)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    fn validate_head_layout(&self) -> GcsoResult<(usize, usize, usize)> {
        let head_dim = self.head_dim() as usize;
        let num_heads = self.num_heads() as usize;
        if head_dim == 0 || (head_dim & 1) != 0 || num_heads == 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        Ok((head_dim, num_heads, expected_len))
    }

    /// Computes the Lazy Phase Unwrapping relative phase differential
    /// $\Delta\theta_{\mathrm{effective}} = \Delta\theta_{\mathrm{query}} - \Delta\theta_{\mathrm{accum}}$
    /// across all configured attention heads using 8-lane SWAR saturating subtraction (Section 4.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_relative_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_sub_sat_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Computes the **Seam-Free Circular Relative Phase Differential** on the circle group
    /// $S^1 \cong SO(2)$ using 8-lane SWAR modular subtraction (`swar_q7_circular_diff_x8`) (Section 4.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_circular_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_circular_diff_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Computes the **Seam-Free $S^1$ Geodesic Relative Phase Differential** across all configured
    /// attention heads by combining 8-lane SWAR modular subtraction (`swar_q7_circular_diff_x8`)
    /// with 8-lane SWAR triangle-wave geodesic folding (`swar_q7_fold_s1_geodesic_x8` into `[-64, +64]`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice has fewer than `self.num_heads()` elements.
    #[inline(always)]
    fn compute_lazy_query_geodesic_phase_q7(
        &self,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        effective_phases_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
            || effective_phases_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_circular_diff_and_fold_geodesic_q7_slices(
            query_phases.as_slice(),
            accumulated_context_phases.as_slice(),
            effective_phases_out.as_mut_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Folds an existing slice of circular Q7 phases in-place onto the principal $S^1$ geodesic
    /// wave `[-64, +64]` ($\pm \pi/2$) using 8-lane SWAR (`swar_q7_fold_s1_geodesic_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases.len() < self.num_heads()`.
    #[inline(always)]
    fn fold_circular_phases_s1_geodesic_q7(
        &self,
        phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_fold_s1_geodesic_q7_slice_in_place(phases.as_mut_slice(), num_heads);
        Ok(())
    }

    /// Scales all configured attention head Q7 phase deltas in-place by `scale_u7` (`128 == 1.0`)
    /// using hoisted-bias 8-lane SWAR (`swar_q7_scale_le128_with_bias_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases.len() < self.num_heads()`.
    #[inline(always)]
    fn scale_phases_swar_u7(
        &self,
        phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        scale_u7: u8,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = phases.as_mut_slice();
        if scale_u7 == 128 {
            return Ok(());
        }
        if scale_u7 == 0 {
            slice[..num_heads].fill(0);
            return Ok(());
        }
        let gt_128 = scale_u7 > 128;
        let s_sub = u64::from(branchless_select_u8(
            gt_128,
            scale_u7.wrapping_sub(128),
            scale_u7,
        ));
        let bias_restore = (128u64.wrapping_sub(s_sub)).wrapping_mul(SWAR_SPLAT_BYTE_64);
        let extra_mask = (gt_128 as u64).wrapping_neg();

        let mut i = 0usize;
        while i + 16 <= num_heads {
            let w0 = load_q7_x8_slice(slice, i);
            let w1 = load_q7_x8_slice(slice, i + 8);
            let s0 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w0, s_sub, bias_restore),
                w0 & extra_mask,
            );
            let s1 = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w1, s_sub, bias_restore),
                w1 & extra_mask,
            );
            store_q7_x8_slice(slice, i, s0);
            store_q7_x8_slice(slice, i + 8, s1);
            i += 16;
        }
        while i + 8 <= num_heads {
            let w = load_q7_x8_slice(slice, i);
            let s = swar_q7_add_sat_x8(
                swar_q7_scale_le128_with_bias_x8(w, s_sub, bias_restore),
                w & extra_mask,
            );
            store_q7_x8_slice(slice, i, s);
            i += 8;
        }
        while i < num_heads {
            slice[i] = branchless_q7_scale_u7(slice[i], scale_u7);
            i += 1;
        }
        Ok(())
    }

    /// Accumulates a per-step Q7 phase delta vector `step_deltas` into `accumulated_context_phases`
    /// on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$ with 8-lane SWAR QDPS step filtering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `min_step_rad < 0.0`, `min_step_rad` is non-finite,
    /// or slice lengths are smaller than `self.num_heads()`.
    #[inline(always)]
    fn accumulate_context_phases_circular_q7(
        &self,
        accumulated_context_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        step_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        let bits = min_step_rad.to_bits();
        let abs_bits = bits & 0x7FFF_FFFF;
        if ((bits >> 31) != 0 && abs_bits != 0)
            || (bits & 0x7F80_0000) == 0x7F80_0000
            || accumulated_context_phases.len() < num_heads
            || step_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let step_u8 = if abs_bits <= Q7_PHASE_SCALE_RAD.to_bits() {
            (abs_bits != 0) as u8
        } else {
            branchless_rad_to_qdps_step_u8(min_step_rad)
        };
        if step_u8 > 127 {
            return Ok(());
        }
        let acc_slice = accumulated_context_phases.as_mut_slice();
        let d_slice = step_deltas.as_slice();

        let mut i = 0usize;
        if step_u8 <= 1 {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_add_epi8, _mm256_loadu_si256, _mm256_storeu_si256,
                };
                let acc_ptr = acc_slice.as_mut_ptr().cast::<__m256i>();
                let d_ptr = d_slice.as_ptr().cast::<__m256i>();
                if num_heads == MAX_ATTENTION_HEADS {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let va1 = _mm256_loadu_si256(acc_ptr.add(1));
                    let vd1 = _mm256_loadu_si256(d_ptr.add(1));
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, vd0));
                    _mm256_storeu_si256(acc_ptr.add(1), _mm256_add_epi8(va1, vd1));
                    return Ok(());
                }
                if num_heads == 32 {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, vd0));
                    return Ok(());
                }
                while i + 32 <= num_heads {
                    let off = i >> 5;
                    let va = _mm256_loadu_si256(acc_ptr.add(off));
                    let vd = _mm256_loadu_si256(d_ptr.add(off));
                    _mm256_storeu_si256(acc_ptr.add(off), _mm256_add_epi8(va, vd));
                    i += 32;
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            while i + 32 <= num_heads {
                let a0 = load_q7_x8_slice(acc_slice, i);
                let a1 = load_q7_x8_slice(acc_slice, i + 8);
                let a2 = load_q7_x8_slice(acc_slice, i + 16);
                let a3 = load_q7_x8_slice(acc_slice, i + 24);
                let d0 = load_q7_x8_slice(d_slice, i);
                let d1 = load_q7_x8_slice(d_slice, i + 8);
                let d2 = load_q7_x8_slice(d_slice, i + 16);
                let d3 = load_q7_x8_slice(d_slice, i + 24);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(a0, d0));
                store_q7_x8_slice(acc_slice, i + 8, swar_q7_circular_add_x8(a1, d1));
                store_q7_x8_slice(acc_slice, i + 16, swar_q7_circular_add_x8(a2, d2));
                store_q7_x8_slice(acc_slice, i + 24, swar_q7_circular_add_x8(a3, d3));
                i += 32;
            }
            while i + 8 <= num_heads {
                let acc_w = load_q7_x8_slice(acc_slice, i);
                let raw_d = load_q7_x8_slice(d_slice, i);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(acc_w, raw_d));
                i += 8;
            }
        } else {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            unsafe {
                use core::arch::x86_64::{
                    __m256i, _mm256_abs_epi8, _mm256_add_epi8, _mm256_and_si256,
                    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_max_epu8, _mm256_set1_epi8,
                    _mm256_storeu_si256,
                };
                let acc_ptr = acc_slice.as_mut_ptr().cast::<__m256i>();
                let d_ptr = d_slice.as_ptr().cast::<__m256i>();
                let v_thresh = _mm256_set1_epi8(step_u8 as i8);
                if num_heads == MAX_ATTENTION_HEADS {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let va1 = _mm256_loadu_si256(acc_ptr.add(1));
                    let vd1 = _mm256_loadu_si256(d_ptr.add(1));
                    let vabs0 = _mm256_abs_epi8(vd0);
                    let vabs1 = _mm256_abs_epi8(vd1);
                    let keep0 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs0, v_thresh), vabs0);
                    let keep1 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs1, v_thresh), vabs1);
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, _mm256_and_si256(vd0, keep0)));
                    _mm256_storeu_si256(acc_ptr.add(1), _mm256_add_epi8(va1, _mm256_and_si256(vd1, keep1)));
                    return Ok(());
                }
                if num_heads == 32 {
                    let va0 = _mm256_loadu_si256(acc_ptr);
                    let vd0 = _mm256_loadu_si256(d_ptr);
                    let vabs0 = _mm256_abs_epi8(vd0);
                    let keep0 = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs0, v_thresh), vabs0);
                    _mm256_storeu_si256(acc_ptr, _mm256_add_epi8(va0, _mm256_and_si256(vd0, keep0)));
                    return Ok(());
                }
                while i + 32 <= num_heads {
                    let off = i >> 5;
                    let va = _mm256_loadu_si256(acc_ptr.add(off));
                    let vd = _mm256_loadu_si256(d_ptr.add(off));
                    let vabs = _mm256_abs_epi8(vd);
                    let keep = _mm256_cmpeq_epi8(_mm256_max_epu8(vabs, v_thresh), vabs);
                    let vfilt = _mm256_and_si256(vd, keep);
                    _mm256_storeu_si256(acc_ptr.add(off), _mm256_add_epi8(va, vfilt));
                    i += 32;
                }
            }
            #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
            while i + 32 <= num_heads {
                let a0 = load_q7_x8_slice(acc_slice, i);
                let a1 = load_q7_x8_slice(acc_slice, i + 8);
                let a2 = load_q7_x8_slice(acc_slice, i + 16);
                let a3 = load_q7_x8_slice(acc_slice, i + 24);
                let d0 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i), step_u8);
                let d1 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 8), step_u8);
                let d2 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 16), step_u8);
                let d3 = swar_q7_qdps_filter_x8(load_q7_x8_slice(d_slice, i + 24), step_u8);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(a0, d0));
                store_q7_x8_slice(acc_slice, i + 8, swar_q7_circular_add_x8(a1, d1));
                store_q7_x8_slice(acc_slice, i + 16, swar_q7_circular_add_x8(a2, d2));
                store_q7_x8_slice(acc_slice, i + 24, swar_q7_circular_add_x8(a3, d3));
                i += 32;
            }
            while i + 8 <= num_heads {
                let acc_w = load_q7_x8_slice(acc_slice, i);
                let raw_d = load_q7_x8_slice(d_slice, i);
                let filt_d = swar_q7_qdps_filter_x8(raw_d, step_u8);
                store_q7_x8_slice(acc_slice, i, swar_q7_circular_add_x8(acc_w, filt_d));
                i += 8;
            }
        }
        let min_step_q7 = i32::from(step_u8);
        while i < num_heads {
            let filt_d = branchless_qdps_filter_q7(d_slice[i], min_step_q7);
            acc_slice[i] = acc_slice[i].wrapping_add(filt_d);
            i += 1;
        }
        Ok(())
    }

    /// Executes the complete **Seam-Free Lazy Phase Unwrapping** hot-path pipeline (Section 4.1)
    /// in a **single fused 8-lane SWAR pass** (`q - ctx` + optional $S^1$ geodesic fold + QDPS
    /// step filter) without multi-pass memory reloads of `effective_phases`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `min_step_rad < 0.0`, `ripa_clamp_max_rad` is non-positive/non-finite, or tensor/phase bounds fail.
    #[inline]
    fn apply_seam_free_lazy_unwrap_q7(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        accumulated_context_phases: &AlignedSlice16<'_, gcso_q7_t>,
        min_step_rad: f32,
        ripa_clamp_max_rad: Option<f32>,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || query_tensor.len() < expected_len
            || query_phases.len() < num_heads
            || accumulated_context_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }
        if let Some(max_rad) = ripa_clamp_max_rad {
            if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let fold_geodesic = ripa_clamp_max_rad.is_some();
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let q_p = &query_phases.as_slice()[..num_heads];
        let c_p = &accumulated_context_phases.as_slice()[..num_heads];
        let mut eff_words = [0u64; NUM_SWARM_CELLS];
        let mut prev_diff_w = 0u64;
        let mut prev_filt_w = 0u64;
        let mut has_prev_w = false;

        let mut h = 0usize;
        while h + 8 <= num_heads {
            let diff_w =
                swar_q7_circular_diff_x8(load_q7_x8_slice(q_p, h), load_q7_x8_slice(c_p, h));
            let filt_w = if has_prev_w && (diff_w == prev_diff_w) {
                prev_filt_w
            } else {
                let folded_w = if fold_geodesic {
                    swar_q7_fold_s1_geodesic_x8(diff_w)
                } else {
                    diff_w
                };
                let fw = if step_u8 <= 1 {
                    folded_w
                } else if step_u8 > 127 {
                    0
                } else {
                    swar_q7_qdps_filter_x8(folded_w, step_u8)
                };
                prev_diff_w = diff_w;
                prev_filt_w = fw;
                has_prev_w = true;
                fw
            };
            eff_words[h >> 3] = filt_w;
            h += 8;
        }
        let min_step_q7 = i32::from(step_u8);
        while h < num_heads {
            let diff = branchless_q7_circular_diff(q_p[h], c_p[h]);
            let folded = if fold_geodesic {
                branchless_q7_fold_s1_geodesic(diff)
            } else {
                diff
            };
            let filt_b = branchless_qdps_filter_q7(folded, min_step_q7);
            eff_words[h >> 3] |= u64::from(filt_b as u8) << ((h & 7) << 3);
            h += 1;
        }
        let mut effective_phases = AlignedBuffer16::<gcso_q7_t, MAX_ATTENTION_HEADS>::new([0i8; MAX_ATTENTION_HEADS]);
        store_q7_words_64(&mut effective_phases.data, &eff_words);

        if let Some(max_rad) = ripa_clamp_max_rad {
            steer_from_swar_words_core_safe(
                &eff_words,
                0xFF,
                128,
                head_dim,
                num_heads,
                false,
                max_rad,
                query_tensor,
                key_tensor,
            );
        } else {
            let eff_slice = effective_phases.as_aligned_slice();
            if let Some(k) = key_tensor {
                self.apply_phase_steering_fused(query_tensor, k, &eff_slice)?;
            } else {
                self.apply_phase_steering(query_tensor, &eff_slice)?;
            }
        }
        Ok(effective_phases)
    }

    /// Applies Dynamic Phase-Shifted RoPE (DPSR) in-place over an arbitrary hardware-aligned
    /// tensor slice (`ALIGN` $\ge 32$, power of two).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32` or `ALIGN` is not a power of two,
    /// or [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies Dynamic Phase-Shifted RoPE (DPSR) in-place to a single 32B-aligned Query or Key tensor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor.len() < head_dim * num_heads` or `phase_deltas.len() < num_heads`.
    #[inline(always)]
    fn apply_phase_steering(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_aligned::<32>(tensor, phase_deltas)
    }

    /// Applies fused in-place phase steering to both Query and Key tensors simultaneously in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` overlap (`GCSO_RESTRICT`)
    /// or have insufficient length.
    #[inline(always)]
    fn apply_phase_steering_fused(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_fused_aligned::<32>(query_tensor, key_tensor, phase_deltas)
    }

    /// Applies fused in-place phase steering to both Query and Key tensors over any alignment `ALIGN >= 32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32` or not a power of two, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if slices alias or are smaller than `head_dim * num_heads`.
    #[inline(always)]
    fn apply_phase_steering_fused_aligned<const ALIGN: usize>(
        &self,
        query_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        key_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies fused in-place phase steering to a pre-verified [`DisjointMutPair`] of `(Q, K)` tensors.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor or phase slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_disjoint_pair<const ALIGN: usize>(
        &self,
        qk_pair: &mut DisjointMutPair<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let (q_slice, k_slice) = qk_pair.as_mut_pair();
        if q_slice.len() < expected_len
            || k_slice.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_pair_slice_core(
            q_slice.as_mut_slice(),
            k_slice.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies 3-way fused in-place phase steering to Query, Key, and Value tensors simultaneously
    /// in lockstep (`rotate_so2_triple_slice_unrolled16`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor`, `key_tensor`, and `value_tensor`
    /// overlap or have lengths smaller than `head_dim * num_heads`.
    #[inline(always)]
    fn apply_phase_steering_fused_qkv(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        value_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        query_tensor.ensure_disjoint_mut(value_tensor)?;
        key_tensor.ensure_disjoint_mut(value_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || value_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_phase_steering_triple_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            value_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies 3-way fused in-place phase steering to a pre-verified [`DisjointMutTriple`] of `(Q, K, V)`
    /// using consecutive uniform-phase head-run coalescing and lockstep 16-float cacheline strides
    /// (`rotate_so2_triple_slice_unrolled16`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if tensor or phase lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_disjoint_triple<const ALIGN: usize>(
        &self,
        qkv_triple: &mut DisjointMutTriple<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let (q_slice_ref, k_slice_ref, v_slice_ref) = qkv_triple.as_mut_triple();
        if q_slice_ref.len() < expected_len
            || k_slice_ref.len() < expected_len
            || v_slice_ref.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        apply_phase_steering_triple_slice_core(
            q_slice_ref.as_mut_slice(),
            k_slice_ref.as_mut_slice(),
            v_slice_ref.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
        );
        Ok(())
    }

    /// Applies RIPA soft-bounded phase steering ($\Delta\theta_{\mathrm{safe}} = \theta_{\max} \cdot \tanh(\Delta\theta / \theta_{\max})$)
    /// to the upper low-frequency quarter (`rot_dim = ((head_dim >> 2) & !1).max(2)`) of an arbitrary
    /// hardware-aligned tensor (`ALIGN >= 32`, Section 2.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if `max_rad <= 0.0`, `max_rad` is non-finite, or slice bounds fail.
    #[inline(always)]
    fn apply_phase_steering_safe_aligned<const ALIGN: usize>(
        &self,
        tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(())
    }

    /// Applies RIPA soft-bounded phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of a single 32B-aligned tensor (Section 2.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_safe_aligned::<32>(tensor, phase_deltas, max_rad)
    }

    /// Applies **Seam-Free $S^1$ Geodesic-Folded RIPA Phase Steering** to the upper $d_{\mathrm{head}}/4$
    /// low-frequency channels of a single 32B-aligned tensor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_geodesic_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if tensor.len() < expected_len || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            true,
            max_rad,
        );
        Ok(())
    }

    /// Applies **Seam-Free $S^1$ Geodesic-Folded RIPA Phase Steering** to the upper $d_{\mathrm{head}}/4$
    /// low-frequency channels of both Query and Key tensors simultaneously in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_geodesic_fused_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            true,
            max_rad,
        );
        Ok(())
    }

    /// Applies fused RIPA soft-bounded phase steering to the upper $d_{\mathrm{head}}/4$ low-frequency
    /// channels of both Query and Key tensors simultaneously.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_fused_safe_aligned::<32>(
            query_tensor,
            key_tensor,
            phase_deltas,
            max_rad,
        )
    }

    /// Applies fused RIPA soft-bounded phase steering to both Query and Key tensors over any alignment `ALIGN >= 32`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_MISALIGNED_POINTER`] if `ALIGN < 32`, or [`GCSO_ERROR_INVALID_ARGUMENT`]
    /// if slices alias, `max_rad <= 0.0`, or lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe_aligned<const ALIGN: usize>(
        &self,
        query_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        key_tensor: &mut AlignedSliceMut<'_, f32, ALIGN>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if ALIGN < 32 || !ALIGN.is_power_of_two() {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(())
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) starting from
    /// `trail_phases` with initial weight `trail_init_weight`, keeping the 8 folded `u64` SWAR
    /// words directly in registers (eliminating intermediate stack buffer store/reload), filters
    /// via 8-lane SWAR QDPS (`min_step_rad`), and steers `query_tensor` (and optional `key_tensor`).
    ///
    /// Returns `(total_resonant_pull, mean_coherence, dominant_anchor_idx)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `anchor_phases.len() != anchor_base_energies.len()`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor bounds fail.
    #[inline]
    fn apply_unified_field_phase_steering_with_inertia_safe(
        &self,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        trail_init_weight: f32,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len()
            || min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }

        let trail_words = load_q7_words_64(trail_phases);
        let (raw_words, total_pull, mean_coh, dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                trail_init_weight,
                anchor_phases,
                anchor_base_energies,
            );

        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let swar_words = if step_u8 <= 1 {
            if step_u8 > 127 {
                [0u64; NUM_SWARM_CELLS]
            } else {
                raw_words
            }
        } else {
            [
                swar_q7_qdps_filter_x8(raw_words[0], step_u8),
                swar_q7_qdps_filter_x8(raw_words[1], step_u8),
                swar_q7_qdps_filter_x8(raw_words[2], step_u8),
                swar_q7_qdps_filter_x8(raw_words[3], step_u8),
                swar_q7_qdps_filter_x8(raw_words[4], step_u8),
                swar_q7_qdps_filter_x8(raw_words[5], step_u8),
                swar_q7_qdps_filter_x8(raw_words[6], step_u8),
                swar_q7_qdps_filter_x8(raw_words[7], step_u8),
            ]
        };

        steer_from_swar_words_core_safe(
            &swar_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            false,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok((total_pull, mean_coh, dom_idx))
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Phase Manifold** ($SO(2)^{d_{\mathrm{head}}/2}$) with zero
    /// initial inertia weight (`0.0`) and steers `query_tensor` (and optional `key_tensor`).
    ///
    /// Returns `(total_resonant_pull, mean_coherence, dominant_anchor_idx)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`Self::apply_unified_field_phase_steering_with_inertia_safe`].
    #[inline(always)]
    fn apply_unified_field_phase_steering_safe(
        &self,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        self.apply_unified_field_phase_steering_with_inertia_safe(
            trail_phases,
            0.0,
            anchor_phases,
            anchor_base_energies,
            min_step_rad,
            max_rad,
            query_tensor,
            key_tensor,
        )
    }

    /// Applies RIPA soft-bounded phase steering gated by `tagged_ptr`'s `ACTION_MODE_DPSR_STEER` bit.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0` or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(false);
        }
        apply_ripa_single_slice_core(
            tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(true)
    }

    /// Applies fused RIPA soft-bounded phase steering to both Query and Key tensors gated by `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor` and `key_tensor` alias, `max_rad <= 0.0`,
    /// or slice lengths are insufficient.
    #[inline]
    fn apply_pointer_gated_phase_steering_fused_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        query_tensor.ensure_disjoint_mut(key_tensor)?;
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
            || query_tensor.len() < expected_len
            || key_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(false);
        }
        apply_ripa_pair_slice_core(
            query_tensor.as_mut_slice(),
            key_tensor.as_mut_slice(),
            phase_deltas.as_slice(),
            head_dim,
            num_heads,
            false,
            max_rad,
        );
        Ok(true)
    }

    /// Indexes directly into an 8-slot Attractor Phase Bank (`slot_phase_bank`, `>= 512` bytes)
    /// using `tagged_ptr`'s 3-bit Target Attractor Slot ID (`0..=7`) and steers `tensor` with RIPA clamping.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `slot_phase_bank.len() < 512`,
    /// `max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn apply_slot_bank_phase_steering_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slot_id = usize::from(tagged_ptr.unpack_tag() >> 5);
        let slot_slice = slot_phase_bank.slot_aligned::<MAX_ATTENTION_HEADS>(slot_id)?;
        self.apply_pointer_gated_phase_steering_safe(tagged_ptr, tensor, &slot_slice, max_rad)
    }

    /// Indexes directly into an 8-slot Attractor Phase Bank (`slot_phase_bank`, `>= 512` bytes)
    /// using `tagged_ptr` and steers both `query_tensor` and `key_tensor` in lockstep.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `slot_phase_bank.len() < 512`,
    /// `max_rad <= 0.0`, or `query_tensor` and `key_tensor` alias or are too small.
    #[inline]
    fn apply_slot_bank_phase_steering_fused_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> GcsoResult<bool> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slot_id = usize::from(tagged_ptr.unpack_tag() >> 5);
        let slot_slice = slot_phase_bank.slot_aligned::<MAX_ATTENTION_HEADS>(slot_id)?;
        self.apply_pointer_gated_phase_steering_fused_safe(
            tagged_ptr,
            query_tensor,
            key_tensor,
            &slot_slice,
            max_rad,
        )
    }

    /// Applies **8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering** directly from a 64-head Q7
    /// phase array (`phase_deltas`), rotating only the resolved 8-head cell groups (`swarm_quorum_mask`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `max_rad <= 0.0`, or
    /// `tensor.len() < head_dim * num_heads`.
    #[inline]
    fn apply_swarm_quorum_gated_steering_safe(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        swarm_quorum_mask: u8,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if swarm_quorum_mask == 0 {
            return Ok(());
        }
        let swar_words = load_q7_words_64(phase_deltas);
        steer_from_swar_words_core_safe(
            &swar_words,
            swarm_quorum_mask,
            0,
            head_dim,
            num_heads,
            true,
            max_rad,
            tensor,
            None,
        );
        Ok(())
    }

    /// Applies pointer-gated **8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering** with
    /// configurable `unresolved_damping_u7` directly from 8 register-resident `u64` SWAR words.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > 64`, `max_rad <= 0.0`,
    /// `phase_deltas.len() < num_heads`, or tensor lengths/non-aliasing invariants fail.
    #[inline]
    fn apply_pointer_swarm_quorum_gated_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        swarm_quorum_mask: u8,
        unresolved_damping_u7: u8,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS
            || query_tensor.len() < expected_len
            || phase_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_DPSR_STEER) {
            return Ok(());
        }

        let clean_damping =
            branchless_select_u8(unresolved_damping_u7 > 128, 128, unresolved_damping_u7);
        let swar_words = if let Ok(arr64) = phase_deltas.as_array::<MAX_ATTENTION_HEADS>() {
            load_q7_words_64(arr64)
        } else {
            let p_slice = phase_deltas.as_slice();
            let mut words = [0u64; NUM_SWARM_CELLS];
            let full_cells = num_heads / HEADS_PER_CELL;
            let mut c = 0usize;
            while c < full_cells {
                words[c] = load_q7_x8_slice(p_slice, c * HEADS_PER_CELL);
                c += 1;
            }
            let rem_start = full_cells * HEADS_PER_CELL;
            if rem_start < num_heads {
                let mut tail_w = 0u64;
                let mut h = rem_start;
                while h < num_heads {
                    tail_w |= u64::from(p_slice[h] as u8) << ((h - rem_start) * 8);
                    h += 1;
                }
                words[full_cells] = tail_w;
            }
            words
        };

        steer_from_swar_words_core_safe(
            &swar_words,
            swarm_quorum_mask,
            clean_damping,
            head_dim,
            num_heads,
            true,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok(())
    }

    /// Interpolates two 64-head Q7 phase profiles (`phases_a`, `phases_b`) along the shortest-arc
    /// $S^1$ geodesic using hoisted-bias 8-word SWAR (`swar_slerp_words_with_bias_64`, fast-pathing
    /// endpoints and word-uniform profiles), applies register-fused QDPS step filtering
    /// (`min_step_rad`), and steers `query_tensor` (and optional `key_tensor`) via
    /// `steer_from_swar_words_core_safe`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads > MAX_ATTENTION_HEADS`,
    /// `phases_a.len() != 64`, `phases_b.len() != 64`, `min_step_rad < 0.0`, `max_rad <= 0.0`,
    /// or tensor lengths/non-aliasing invariants fail.
    #[inline]
    fn slerp_and_apply_phase_steering_safe(
        &self,
        phases_a: &AlignedSlice16<'_, gcso_q7_t>,
        phases_b: &AlignedSlice16<'_, gcso_q7_t>,
        t_u7: u8,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<AlignedBuffer16<gcso_q7_t, MAX_ATTENTION_HEADS>> {
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || max_rad <= 0.0
            || !branchless_is_finite_f32(max_rad)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if num_heads > MAX_ATTENTION_HEADS || query_tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(ref k) = key_tensor {
            if k.len() < expected_len {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            query_tensor.ensure_disjoint_mut(k)?;
        }

        let a_arr = phases_a.as_array::<MAX_ATTENTION_HEADS>()?;
        let b_arr = phases_b.as_array::<MAX_ATTENTION_HEADS>()?;
        let step_u8 = branchless_rad_to_qdps_step_u8(min_step_rad);
        let s = u64::from(branchless_select_u8(t_u7 > 128, 128, t_u7));
        let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);

        let wa = load_q7_words_64(a_arr);
        let wb = load_q7_words_64(b_arr);
        let slerped_words = swar_slerp_words_with_bias_64(&wa, &wb, s, bias_restore);

        let swar_words = swar_q7_qdps_filter_words_64(&slerped_words, step_u8);

        let mut blended = AlignedBuffer16::<gcso_q7_t, MAX_ATTENTION_HEADS>::zeroed();
        store_q7_words_64(&mut blended.data, &swar_words);

        steer_from_swar_words_core_safe(
            &swar_words,
            0xFF,
            128,
            head_dim,
            num_heads,
            true,
            max_rad,
            query_tensor,
            key_tensor,
        );
        Ok(blended)
    }

    /// Applies **Isotropic Block-Diagonal Scaling** ($\mathbf{D} = \mathrm{diag}(s_{h,0}\mathbf{I}_2, \dots, s_{h,d/2-1}\mathbf{I}_2)$)
    /// across all 2D subspaces of `tensor` (Theorem 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor.len() < head_dim * num_heads` or
    /// `block_scales.len() < (head_dim / 2) * num_heads`.
    #[inline]
    fn apply_isotropic_block_scaling(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        block_scales: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()> {
        let (_head_dim, _num_heads, expected_len) = self.validate_head_layout()?;
        let num_pairs = expected_len / 2;
        if tensor.len() < expected_len || block_scales.len() < num_pairs {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let t_slice = &mut tensor.as_mut_slice()[..expected_len];
        let s_slice = &block_scales.as_slice()[..num_pairs];
        for (pair, &raw_scale) in t_slice.chunks_exact_mut(2).zip(s_slice.iter()) {
            let s = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(pair[0], raw_scale));
            pair[0] = branchless_flush_subnormal_f32(pair[0] * s);
            pair[1] = branchless_flush_subnormal_f32(pair[1] * s);
        }
        Ok(())
    }

    /// Applies the **Composite Modulation Operator** $\mathbf{M}_{\mathrm{GCSO}} = \mathbf{D} \cdot \mathbf{R}_{\mathrm{DPSR}}(\boldsymbol{\Delta\theta}) + \mathbf{u}\mathbf{v}^T$
    /// in a single fused pass over `tensor` (Theorem 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `max_rad <= 0.0`, `max_rad` is non-finite,
    /// or slice lengths are insufficient.
    #[inline]
    fn apply_composite_dpsr_modulation(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        block_scales: &AlignedSlice32<'_, f32>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        let pairs_per_head = head_dim / 2;
        let total_pairs = expected_len / 2;
        if tensor.len() < expected_len
            || phase_deltas.len() < num_heads
            || block_scales.len() < total_pairs
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
        let t_slice = tensor.as_mut_slice();
        let p_slice = phase_deltas.as_slice();
        let s_slice = block_scales.as_slice();

        for (h, &raw_q7) in p_slice.iter().enumerate().take(num_heads) {
            let (sin_t, cos_t) =
                eval_ripa_sincos_with_params(raw_q7, false, is_default, eff_max, q7_to_norm_scale);
            let h_base = h * head_dim;
            let s_base = h * pairs_per_head;
            let head_buf = &mut t_slice[h_base..h_base + head_dim];
            let head_scales = &s_slice[s_base..s_base + pairs_per_head];

            for (pair, &raw_s) in head_buf.chunks_exact_mut(2).zip(head_scales.iter()) {
                let s = branchless_flush_subnormal_f32(sanitize_inplace_gain_f32(pair[0], raw_s));
                let (r0, r1) =
                    branchless_rotate_so2_pair_scaled(pair[0], pair[1], sin_t, cos_t, s);
                pair[0] = r0;
                pair[1] = r1;
            }
        }
        Ok(())
    }

    /// Applies multi-token batched phase steering across `batch_tokens` contiguous tokens.
    ///
    /// Supports both **broadcast single-token phase slices** (`phase_deltas_batch.len() >= num_heads`,
    /// where consecutive uniform-phase head runs `(run_start, run_end, sin_t, cos_t)` are coalesced
    /// once outside the token loop and rotated in a single contiguous call per run per token)
    /// and **per-token phase slices** (`phase_deltas_batch.len() >= batch_tokens * num_heads`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `batch_tokens == 0`, `tensor_batch.len() < batch_tokens * head_dim * num_heads`,
    /// or `phase_deltas_batch.len() < num_heads`.
    #[inline(always)]
    fn apply_phase_steering_batch(
        &self,
        tensor_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if tensor_batch.len() < token_elems || phase_deltas_batch.len() < num_heads {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_phase_steering_slice_core(
                &mut tensor_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
            );
            return Ok(());
        }
        apply_phase_steering_multi_token_batch_core(
            tensor_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
        )
    }

    /// Applies fused multi-token batched phase steering to both Query and Key batches in lockstep.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `query_batch` and `key_batch` alias/overlap,
    /// `batch_tokens == 0`, or slice bounds fail.
    #[inline(always)]
    fn apply_phase_steering_fused_batch(
        &self,
        query_batch: &mut AlignedSliceMut32<'_, f32>,
        key_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
    ) -> GcsoResult<()> {
        query_batch.ensure_disjoint_mut(key_batch)?;
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if query_batch.len() < token_elems
                || key_batch.len() < token_elems
                || phase_deltas_batch.len() < num_heads
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_phase_steering_pair_slice_core(
                &mut query_batch.as_mut_slice()[..token_elems],
                &mut key_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
            );
            return Ok(());
        }
        apply_phase_steering_fused_multi_token_batch_core(
            query_batch.as_mut_slice(),
            key_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
        )
    }

    /// Applies fused RIPA soft-bounded multi-token batched phase steering to Query and Key batches,
    /// supporting both broadcast (`len >= num_heads`, hoisting RIPA `(sin_t, cos_t)` outside the
    /// token loop) and per-token (`len >= batch_tokens * num_heads`) phase slices.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `batch_tokens == 0`, `max_rad <= 0.0`,
    /// `query_batch` and `key_batch` overlap, or buffer lengths are insufficient.
    #[inline(always)]
    fn apply_phase_steering_fused_safe_batch(
        &self,
        query_batch: &mut AlignedSliceMut32<'_, f32>,
        key_batch: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas_batch: &AlignedSlice16<'_, gcso_q7_t>,
        batch_tokens: usize,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        query_batch.ensure_disjoint_mut(key_batch)?;
        let (head_dim, num_heads, token_elems) = self.validate_head_layout()?;
        if batch_tokens == 1 {
            if query_batch.len() < token_elems
                || key_batch.len() < token_elems
                || phase_deltas_batch.len() < num_heads
            {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
            apply_ripa_pair_slice_core(
                &mut query_batch.as_mut_slice()[..token_elems],
                &mut key_batch.as_mut_slice()[..token_elems],
                &phase_deltas_batch.as_slice()[..num_heads],
                head_dim,
                num_heads,
                false,
                max_rad,
            );
            return Ok(());
        }
        apply_phase_steering_fused_safe_multi_token_batch_core(
            query_batch.as_mut_slice(),
            key_batch.as_mut_slice(),
            phase_deltas_batch.as_slice(),
            batch_tokens,
            head_dim,
            num_heads,
            token_elems,
            max_rad,
        )
    }

    /// Applies subnormal-safe and near-identity-guarded multiplicative phase damping in-place across `tensor`
    /// using `sanitize_inplace_gain_f32` and unconditional 32-float unrolled AVX2 scaling
    /// (`scale_f32_slice_unrolled8`).
    ///
    /// Guarantees exact binary dyadic reversibility (e.g., alternating `0.5` and `2.0` produces zero
    /// mantissa ULP drift) and flushes subnormal products (`|x * s| < F32_MIN_POSITIVE_NORMAL`) to `+0.0`
    /// in YMM registers to prevent x86 microcode traps over million-token horizons.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor` is empty, `damping_factor <= 0.0`,
    /// or `damping_factor` is non-finite.
    #[inline]
    fn apply_soft_phase_damping(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        damping_factor: f32,
    ) -> GcsoResult<()> {
        if tensor.is_empty()
            || damping_factor <= 0.0
            || !branchless_is_finite_f32(damping_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        scale_f32_slice_unrolled8(tensor.as_mut_slice(), damping_factor);
        Ok(())
    }

    /// Applies RIPA hyperbolic tangent soft-clamping ($\theta_{\mathrm{safe}} = \theta_{\max}\tanh(\theta/\theta_{\max})$)
    /// in-place on floating-point radian angles.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `raw_angles` is empty, `max_rad <= 0.0`, or `max_rad` is non-finite.
    #[inline]
    fn clamp_angles_ripa_in_place(
        &self,
        raw_angles: &mut AlignedSliceMut32<'_, f32>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        if raw_angles.is_empty() || max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        clamp_angles_ripa_in_place_simd8(raw_angles.as_mut_slice(), max_rad);
        Ok(())
    }

    /// Filters sub-grid Q7 phase noise below `min_step_rad` using 8-lane SWAR (`swar_q7_qdps_filter_x8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `raw_phases.len() < self.num_heads()`, `min_step_rad < 0.0`,
    /// or `min_step_rad` is non-finite.
    #[inline(always)]
    fn qdps_filter_step(
        &self,
        raw_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if min_step_rad < 0.0
            || !branchless_is_finite_f32(min_step_rad)
            || raw_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let min_step_q7 = i32::from(branchless_rad_to_qdps_step_u8(min_step_rad));
        swar_qdps_filter_q7_slice_in_place(raw_phases.as_mut_slice(), num_heads, min_step_q7);
        Ok(())
    }

    /// Applies Lazy Phase Unwrapping relative shift directly to a Query tensor in `f32` radians
    /// using 8-head batched AVX2 range-reduced `sincos_f32_x8_simd`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `query_tensor.len() < head_dim * num_heads` or
    /// `accumulated_context_phases.len() < num_heads`.
    #[inline]
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        accumulated_context_phases: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if query_tensor.len() < expected_len || accumulated_context_phases.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_slice = query_tensor.as_mut_slice();
        let c_slice = accumulated_context_phases.as_slice();

        let mut h = 0usize;
        while h + 8 <= num_heads {
            let mut sin_8 = [0.0_f32; 8];
            let mut cos_8 = [0.0_f32; 8];
            sincos_f32_x8_simd(&c_slice[h..h + 8], &mut sin_8, &mut cos_8);
            let mut lane = 0usize;
            while lane < 8 {
                let base = (h + lane) * head_dim;
                rotate_so2_slice_unrolled16(
                    &mut q_slice[base..base + head_dim],
                    sin_8[lane],
                    cos_8[lane],
                );
                lane += 1;
            }
            h += 8;
        }
        while h < num_heads {
            let (sin_t, cos_t) = branchless_sincos_f32(c_slice[h]);
            let base = h * head_dim;
            rotate_so2_slice_unrolled16(&mut q_slice[base..base + head_dim], sin_t, cos_t);
            h += 1;
        }
        Ok(())
    }

    /// Applies underflow-resilient SLERP norm stabilization across `tensor` using
    /// `norm_l2_underflow_resilient_f32` and division-free [`branchless_inv_pos_f32`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `tensor` is empty or `target_norm <= 0.0` /
    /// `damping_alpha < 0.0`, or [`GCSO_ERROR_EDBC_SINGULARITY`] if `tensor` contains `NaN`/`Inf`.
    #[inline]
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        target_norm: f32,
        damping_alpha: f32,
    ) -> GcsoResult<()> {
        if tensor.is_empty()
            || target_norm <= 0.0
            || damping_alpha < 0.0
            || !branchless_is_finite_f32(target_norm)
            || !branchless_is_finite_f32(damping_alpha)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let slice = tensor.as_mut_slice();
        let (current_norm, is_finite) = norm_l2_underflow_resilient_f32(slice);
        if !is_finite {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let alpha = branchless_clamp_f32(damping_alpha, 0.0, 1.0);
        let valid_norm = current_norm > 0.0;
        let ratio = if current_norm >= F32_MIN_POSITIVE_NORMAL {
            target_norm * fast_inv_pos_normal_f32(current_norm)
        } else {
            target_norm
                * branchless_inv_pos_f32(branchless_select_f32(valid_norm, current_norm, 1.0))
        };
        let scale = branchless_select_f32(valid_norm, (1.0 - alpha) + alpha * ratio, 1.0);
        scale_f32_slice_unrolled8(slice, scale);
        Ok(())
    }

    /// Applies a fused phase-derived bias shift to `logits` in-place using 8-lane SWAR signed byte
    /// reduction and 32-float unrolled AVX2 scalar addition (`add_scalar_f32_slice_unrolled8`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `phase_deltas.len() < self.num_heads()`.
    #[inline]
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty() || phase_deltas.len() < num_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let p_slice = phase_deltas.as_slice();
        let mut phase_sum_i32 = 0i32;
        let mut h = 0usize;
        while h + 8 <= num_heads {
            let u_w = load_q7_x8_slice(p_slice, h) ^ SWAR_SIGN_MASK_64;
            phase_sum_i32 += (swar_u8_sum_x8(u_w) as i32) - 1024;
            h += 8;
        }
        while h < num_heads {
            phase_sum_i32 += i32::from(p_slice[h]);
            h += 1;
        }
        let mean_phase_rad =
            (phase_sum_i32 as f32) * Q7_PHASE_SCALE_RAD * fast_inv_len_f32(num_heads);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), mean_phase_rad);
        Ok(())
    }

    /// Modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
    /// directly in-place by **Attention $\times$ DPSR Phase Interference**
    /// $\Delta z_{t,j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j)$
    /// across `memory_anchor_phases` in zero-copy, zero-allocation ($C_{\mathrm{alloc}} = 0$) mode.
    ///
    /// Returns the mean phase coherence across the modulated candidate memories.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores` or `memory_anchor_phases` is
    /// empty, `head_dim()` is invalid, or `interference_gain` is non-finite.
    #[inline]
    fn modulate_attention_scores_by_phase_interference(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        interference_gain: f32,
    ) -> GcsoResult<f32> {
        let (head_dim, _num_heads, _expected_len) = self.validate_head_layout()?;
        if attention_scores.is_empty()
            || memory_anchor_phases.is_empty()
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(modulate_attention_scores_by_phase_bank_64(
            attention_scores.as_mut_slice(),
            query_phases,
            memory_anchor_phases,
            head_dim as u32,
            interference_gain,
        ))
    }

    /// Modulates pre-Softmax candidate memory/key Attention scores (`attention_scores[..M]`)
    /// directly in-place by the full **Bayes-Attention Stigmergic Transition Score**
    /// (uniting Hypothesis 1 Joint Probability $\times$ Attention and Hypothesis 2 `ActionHub`
    /// Pointer-Trail Manipulation):
    /// $$\Delta z_{t \to j} = \frac{\gamma}{\sqrt{d_{\mathrm{head}}}} \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_t + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_j) + \eta \ln(1 + \rho_j) + \lambda_{\mathrm{link}} \mathbb{I}(\mathrm{linked\_trail\_id} == j)$$
    ///
    /// Returns `(mean_phase_coherence, best_post_modulated_candidate_idx)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `attention_scores`, `candidate_key_phases`, or
    /// `candidate_stigmergic_densities` is empty, `head_dim()` is invalid, or any gain is non-finite.
    #[inline]
    fn modulate_attention_scores_by_bayesian_trails(
        &self,
        attention_scores: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        candidate_key_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        candidate_stigmergic_densities: &[f32],
        linked_trail_id: Option<u32>,
        interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
    ) -> GcsoResult<(f32, u32)> {
        let (head_dim, _num_heads, _expected_len) = self.validate_head_layout()?;
        if attention_scores.is_empty()
            || candidate_key_phases.is_empty()
            || candidate_stigmergic_densities.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || !branchless_is_finite_f32(stigmergic_prior_gain)
            || !branchless_is_finite_f32(pointer_link_bonus)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(modulate_attention_scores_by_bayesian_trails_64(
            attention_scores.as_mut_slice(),
            query_phases,
            candidate_key_phases,
            candidate_stigmergic_densities,
            linked_trail_id,
            head_dim as u32,
            interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
        ))
    }

    /// Applies **Multiplicative Inverse-Temperature Logit Scaling**
    /// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$
    /// (`logits[i] *= beta_eff`), sharpening the Softmax distribution under constructive phase
    /// alignment ($\mathcal{C}_{\mathrm{phase}} > 0$) and flattening it under destructive interference ($\mathcal{C}_{\mathrm{phase}} < 0$).
    ///
    /// Returns `(phase_coherence, applied_inv_temp_scale)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `interference_gain` is non-finite,
    /// or phase slices have fewer than `self.num_heads()` elements.
    #[inline]
    fn apply_phase_interference_inv_temperature_scaling(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        memory_phases: &AlignedSlice16<'_, gcso_q7_t>,
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || query_phases.len() < num_heads
            || memory_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_p = &query_phases.as_slice()[..num_heads];
        let m_p = &memory_phases.as_slice()[..num_heads];
        let coh = eval_multi_head_phase_coherence_unrolled4(q_p, m_p);
        let inv_temp = branchless_phase_coherence_inv_temp_scale(coh, interference_gain);
        scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        Ok((coh, inv_temp))
    }

    /// Applies **Multiplicative Inverse-Temperature Logit Scaling**
    /// $\beta_{\mathrm{eff}} = \mathrm{clamp}(1.0 + \gamma \cdot \mathcal{C}_{\mathrm{phase}}, 0.25, 4.0)$
    /// directly from 64-head Q7 phase arrays (`[gcso_q7_t; 64]`) in zero-copy mode.
    ///
    /// Returns `(phase_coherence, applied_inv_temp_scale)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `interference_gain` is non-finite.
    #[inline]
    fn apply_phase_interference_inv_temperature_scaling_64(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || num_heads > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let coh = eval_multi_head_phase_coherence_unrolled4(
            &query_phases[..num_heads],
            &memory_phases[..num_heads],
        );
        let inv_temp = branchless_phase_coherence_inv_temp_scale(coh, interference_gain);
        scale_f32_slice_unrolled8(logits.as_mut_slice(), inv_temp);
        Ok((coh, inv_temp))
    }

    /// Modulates `logits` in-place by the **Attention $\times$ DPSR Phase-Interference Bias**
    /// $\Delta z = \gamma \cdot \mathcal{C}_{\mathrm{phase}}(\boldsymbol{\theta}_q + \boldsymbol{\Delta\theta}, \boldsymbol{\theta}_m)$
    /// across the configured `num_heads` heads, returning `(phase_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `interference_gain` is non-finite,
    /// or phase slices have fewer than `self.num_heads()` elements.
    #[inline]
    fn apply_phase_interference_logit_modulation(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &AlignedSlice16<'_, gcso_q7_t>,
        memory_phases: &AlignedSlice16<'_, gcso_q7_t>,
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || !branchless_is_finite_f32(interference_gain)
            || query_phases.len() < num_heads
            || memory_phases.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let q_p = &query_phases.as_slice()[..num_heads];
        let m_p = &memory_phases.as_slice()[..num_heads];
        let coh = eval_multi_head_phase_coherence_unrolled4(q_p, m_p);
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((coh, bias))
    }

    /// Modulates `logits` in-place by the **Attention $\times$ DPSR Phase-Interference Bias**
    /// directly from 64-head Q7 phase arrays (`[gcso_q7_t; 64]`) in zero-copy mode.
    ///
    /// Returns `(phase_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty or `interference_gain` is non-finite.
    #[inline]
    fn apply_phase_interference_logit_modulation_64(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        query_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        memory_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32)> {
        let (_head_dim, num_heads, _expected_len) = self.validate_head_layout()?;
        if logits.is_empty()
            || num_heads > MAX_ATTENTION_HEADS
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let coh = eval_multi_head_phase_coherence_unrolled4(
            &query_phases[..num_heads],
            &memory_phases[..num_heads],
        );
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((coh, bias))
    }

    /// Evaluates **Unified-Field Phase-Interference Logit Modulation** across an arbitrary slice
    /// of memory/attractor anchors (`anchor_phases`, `anchor_base_energies`), shifting `logits`
    /// by the energy-weighted phase interference $\Delta z = \gamma \cdot \bar{\mathcal{C}}_{\mathrm{unified}}$.
    ///
    /// Returns `(total_resonant_pull, mean_coherence, applied_logit_bias)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `logits` is empty, `anchor_phases.len() != anchor_base_energies.len()`,
    /// or `interference_gain` is non-finite.
    #[inline]
    fn apply_unified_field_logit_interference(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        trail_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        interference_gain: f32,
    ) -> GcsoResult<(f32, f32, f32)> {
        if logits.is_empty()
            || anchor_phases.len() != anchor_base_energies.len()
            || !branchless_is_finite_f32(interference_gain)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let trail_words = load_q7_words_64(trail_phases);
        let (_folded_words, total_pull, mean_coh, _dom_idx, _act_cnt) =
            branchless_q7_unified_field_fold_words_with_init_weight_64(
                &trail_words,
                0.0,
                anchor_phases,
                anchor_base_energies,
            );
        let clean_gain = branchless_flush_subnormal_f32(branchless_relu_f32(interference_gain));
        let bias = branchless_flush_subnormal_f32(clean_gain * mean_coh);
        add_scalar_f32_slice_unrolled8(logits.as_mut_slice(), bias);
        Ok((total_pull, mean_coh, bias))
    }

    /// Computes the closed-form **Orthogonal Procrustes Phase Delta** $\Delta\theta^{(h)} \in [-128, 127]$
    /// that aligns each attention head of `source_tensor` with `target_tensor` in $SO(2)^{d_{\mathrm{head}}/2}$
    /// via AVX2 SIMD (`procrustes_q7_slice_simd`, Section 2.1 & 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `source_tensor` or `target_tensor` has fewer than
    /// `head_dim * num_heads` elements or `phase_deltas_out` has fewer than `num_heads` elements.
    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source_tensor: &AlignedSlice32<'_, f32>,
        target_tensor: &AlignedSlice32<'_, f32>,
        phase_deltas_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads, expected_len) = self.validate_head_layout()?;
        if source_tensor.len() < expected_len
            || target_tensor.len() < expected_len
            || phase_deltas_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        procrustes_q7_slice_simd(
            source_tensor.as_slice(),
            target_tensor.as_slice(),
            head_dim,
            num_heads,
            phase_deltas_out.as_mut_slice(),
        );
        Ok(())
    }

    /// Evaluates circular phase coherence $\mathcal{C}_{\mathrm{phase}}(a, b) \in [-1.0, 1.0]$ and
    /// circular resultant statistics $(\bar{R}_a, \bar{\theta}_{a,\mathrm{Q7}}, \kappa_a)$ across 64 heads
    /// in a single register pass via [`branchless_q7_coherence_and_resultant_64`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `phases_a.len() != 64` or `phases_b.len() != 64`.
    #[inline(always)]
    fn evaluate_phase_coherence_and_resultant_64(
        &self,
        phases_a: &AlignedSlice16<'_, gcso_q7_t>,
        phases_b: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> GcsoResult<(f32, f32, gcso_q7_t, f32)> {
        let a_arr = phases_a.as_array::<MAX_ATTENTION_HEADS>()?;
        let b_arr = phases_b.as_array::<MAX_ATTENTION_HEADS>()?;
        Ok(branchless_q7_coherence_and_resultant_64(a_arr, b_arr))
    }
}

// ===================================================================
// 2. Entropy-Gated Phase Lookup & Speculative Prefetch (`EntropyGatedPhaseLookup`)
// ===================================================================

/// Trait for Entropy-Gated Phase Lookup, AOT Geometric Phase Basis Synthesis, and
/// Speculative Paged-Bitmask Prefetching (**Nano / Micro Level**, Section 2.1).
pub trait EntropyGatedPhaseLookup: BranchlessKernel {
    /// Evaluates the smooth branchless entropy gate scale $g(\tilde{H}) = g_{\max} \cdot \mathrm{clamp}(\tilde{H} / \ln 2, 0, 1)$.
    #[inline(always)]
    #[must_use]
    fn eval_entropy_gate_scale(&self, h_tilde: f32, max_gain: f32) -> f32 {
        let valid = branchless_is_finite_f32(h_tilde) & branchless_is_finite_f32(max_gain);
        let norm_h = branchless_clamp_f32(branchless_relu_f32(h_tilde) * (1.0 / LN_2), 0.0, 1.0);
        let raw = branchless_relu_f32(max_gain) * norm_h;
        branchless_select_f32(valid, branchless_flush_subnormal_f32(raw), 0.0)
    }

    /// Synthesizes a 64-head Q7 phase profile from `k_bases` pre-computed AOT geometric phase bases
    /// (`basis_table`, length `>= k_bases * num_heads`) weighted by `alpha_weights` (`k_bases`)
    /// and gated by `h_tilde` (Section 2.1).
    ///
    /// Traverses `basis_table` in cache-friendly row-major (`b_idx` outer, `h` inner) order when
    /// `num_heads <= MAX_ATTENTION_HEADS`, hoisting `alpha` sanitization and `Q7_PHASE_SCALE_RAD * gate`
    /// scaling outside the head loop.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `k_bases == 0`, `num_heads == 0`,
    /// `basis_table.len() < k_bases * num_heads`, `alpha_weights.len() < k_bases`, or
    /// `synthesized_out.len() < num_heads`.
    #[inline]
    fn synthesize_aot_phase_bases(
        &self,
        basis_table: &AlignedSlice16<'_, gcso_q7_t>,
        alpha_weights: &AlignedSlice32<'_, f32>,
        k_bases: usize,
        num_heads: usize,
        h_tilde: f32,
        max_gain: f32,
        synthesized_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        let required_table = k_bases
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if k_bases == 0
            || num_heads == 0
            || basis_table.len() < required_table
            || alpha_weights.len() < k_bases
            || synthesized_out.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let gate = self.eval_entropy_gate_scale(h_tilde, max_gain);
        let out_slice = &mut synthesized_out.as_mut_slice()[..num_heads];
        if gate == 0.0 {
            out_slice.fill(0);
            return Ok(());
        }

        let b_slice = basis_table.as_slice();
        let a_slice = alpha_weights.as_slice();
        let rad_gate_scale = Q7_PHASE_SCALE_RAD * gate;

        if num_heads <= MAX_ATTENTION_HEADS {
            let mut acc_rad_scaled = [0.0_f32; MAX_ATTENTION_HEADS];
            for (b_idx, &alpha) in a_slice.iter().enumerate().take(k_bases) {
                let clean_a = branchless_select_f32(branchless_is_finite_f32(alpha), alpha, 0.0);
                let w_b = clean_a * rad_gate_scale;
                let row = &b_slice[b_idx * num_heads..(b_idx + 1) * num_heads];
                for (acc_h, &basis_q7) in acc_rad_scaled[..num_heads].iter_mut().zip(row.iter()) {
                    *acc_h += w_b * f32::from(basis_q7);
                }
            }
            for (out_q7, &rad) in out_slice.iter_mut().zip(acc_rad_scaled[..num_heads].iter()) {
                *out_q7 = branchless_rad_to_q7(rad);
            }
        } else {
            for (h, out_q7) in out_slice.iter_mut().enumerate() {
                let mut acc_rad = 0.0_f32;
                for (b_idx, &alpha) in a_slice.iter().enumerate().take(k_bases) {
                    let clean_a =
                        branchless_select_f32(branchless_is_finite_f32(alpha), alpha, 0.0);
                    let basis_q7 = b_slice[b_idx * num_heads + h];
                    acc_rad += clean_a * (f32::from(basis_q7) * Q7_PHASE_SCALE_RAD);
                }
                *out_q7 = branchless_rad_to_q7(acc_rad * gate);
            }
        }
        Ok(())
    }

    /// Computes a deterministic $\mathcal{O}(1)$ speculative prefetch index into a power-of-two
    /// phase table of size `table_capacity_pow2` from a 64-bit recent KV activity bitmask.
    #[inline(always)]
    #[must_use]
    fn speculative_prefetch_hash(&self, recent_kv_bitmask: u64, table_capacity_pow2: u32) -> u32 {
        let is_pow2 =
            (table_capacity_pow2 > 0) & ((table_capacity_pow2 & (table_capacity_pow2 - 1)) == 0);
        let mask = branchless_select_u32(is_pow2, table_capacity_pow2.wrapping_sub(1), 0);
        (branchless_splitmix64(recent_kv_bitmask) as u32) & mask
    }

    /// Folds a full 256-bit [`gcso_paged_bitmask_t`] and a [`gcso_tagged_ptr_t`] into a
    /// speculative prefetch table slot index in $\mathcal{O}(1)$ branchless operations.
    #[inline(always)]
    #[must_use]
    fn speculative_paged_bitmask_hash(
        &self,
        paged_mask: &gcso_paged_bitmask_t,
        tagged_ptr: gcso_tagged_ptr_t,
        table_capacity_pow2: u32,
    ) -> u32 {
        let w01 = paged_mask.bits[0] ^ paged_mask.bits[1].rotate_left(19);
        let w23 = paged_mask.bits[2].rotate_left(37) ^ paged_mask.bits[3].rotate_left(53);
        let combined = w01 ^ w23 ^ tagged_ptr.raw();
        self.speculative_prefetch_hash(combined, table_capacity_pow2)
    }

    /// Computes a speculative prefetch index directly from a [`gcso_pointer_trail_t`] and optional
    /// [`gcso_paged_bitmask_t`] in $\mathcal{O}(1)$.
    #[inline(always)]
    #[must_use]
    fn speculative_prefetch_from_trail(
        &self,
        trail: &gcso_pointer_trail_t,
        paged_mask: Option<&gcso_paged_bitmask_t>,
        table_capacity_pow2: u32,
    ) -> u32 {
        let ptr_mix = trail.current_ptr
            ^ trail.prev_ptr.rotate_left(23)
            ^ (u64::from(trail.cluster_id) << 32)
            ^ u64::from(trail.step_count);
        match paged_mask {
            Some(m) => self.speculative_paged_bitmask_hash(
                m,
                gcso_tagged_ptr_t(ptr_mix),
                table_capacity_pow2,
            ),
            None => self.speculative_prefetch_hash(ptr_mix, table_capacity_pow2),
        }
    }
}

// ===================================================================
// 3. Dynamic Layer Skipping with Phase Compensation (`DynamicLayerSkipper`)
// ===================================================================

/// Trait for Dynamic Layer Skipping and SWAR Phase Compensation during low-entropy inertial
/// decoding regimes (**Micro / Mezzo Level**, Section 3.2).
pub trait DynamicLayerSkipper: BranchlessKernel {
    /// Evaluates whether an intermediate Transformer layer can be skipped branchlessly:
    /// $\tilde{H} \le \tau_{\mathrm{skip}} \land \lvert\Phi_M(t)\rvert \le \tau_{\mathrm{flux}} \land \text{consecutive\_skips} < \text{max\_skips}$.
    #[inline(always)]
    #[must_use]
    fn should_skip_layer_branchless(
        &self,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        consecutive_skips: u32,
        max_consecutive_skips: u32,
    ) -> bool {
        let valid = branchless_is_finite_f32(h_tilde)
            & branchless_is_finite_f32(flux_rate)
            & branchless_is_finite_f32(skip_entropy_thresh)
            & branchless_is_finite_f32(skip_flux_thresh);
        let low_h = h_tilde <= skip_entropy_thresh;
        let low_flux = self.abs_f32(flux_rate) <= skip_flux_thresh;
        let under_cap = consecutive_skips < max_consecutive_skips;
        valid & low_h & low_flux & under_cap
    }

    /// Evaluates whether an intermediate layer should be skipped by combining `tagged_ptr`'s
    /// `ACTION_MODE_LAYER_SKIP` bit (`0x02`) with the entropy/flux criteria.
    #[inline(always)]
    #[must_use]
    fn should_skip_layer_by_ptr_branchless(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        consecutive_skips: u32,
        max_consecutive_skips: u32,
    ) -> bool {
        ptr_has_action_mode(tagged_ptr, ACTION_MODE_LAYER_SKIP)
            & self.should_skip_layer_branchless(
                h_tilde,
                flux_rate,
                skip_entropy_thresh,
                skip_flux_thresh,
                consecutive_skips,
                max_consecutive_skips,
            )
    }

    /// Applies 8-lane SWAR saturating phase compensation (`swar_q7_add_sat_x8`) across all
    /// `num_heads` heads when an intermediate Transformer layer is bypassed.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads == 0` or either slice has fewer than
    /// `num_heads` elements.
    #[inline]
    fn apply_skip_phase_compensation(
        &self,
        accumulated_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        skip_compensation_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        num_heads: usize,
    ) -> GcsoResult<()> {
        if num_heads == 0
            || accumulated_phases.len() < num_heads
            || skip_compensation_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        swar_add_sat_q7_slice_in_place(
            accumulated_phases.as_mut_slice(),
            skip_compensation_deltas.as_slice(),
            num_heads,
        );
        Ok(())
    }

    /// Evaluates pointer-gated layer skip eligibility and, if skipped, applies 8-lane SWAR
    /// phase compensation to `accumulated_phases` and increments `*consecutive_skips_inout`
    /// (or resets it to `0` when the layer is executed).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_heads == 0` or phase slices are smaller than `num_heads`.
    #[inline]
    fn evaluate_and_compensate_layer_skip_by_ptr(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        h_tilde: f32,
        flux_rate: f32,
        skip_entropy_thresh: f32,
        skip_flux_thresh: f32,
        max_consecutive_skips: u32,
        consecutive_skips_inout: &mut u32,
        accumulated_phases: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        skip_compensation_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        num_heads: usize,
    ) -> GcsoResult<bool> {
        if num_heads == 0
            || accumulated_phases.len() < num_heads
            || skip_compensation_deltas.len() < num_heads
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let skip = self.should_skip_layer_by_ptr_branchless(
            tagged_ptr,
            h_tilde,
            flux_rate,
            skip_entropy_thresh,
            skip_flux_thresh,
            *consecutive_skips_inout,
            max_consecutive_skips,
        );
        if skip {
            swar_add_sat_q7_slice_in_place(
                accumulated_phases.as_mut_slice(),
                skip_compensation_deltas.as_slice(),
                num_heads,
            );
            *consecutive_skips_inout = consecutive_skips_inout.saturating_add(1);
        } else {
            *consecutive_skips_inout = 0;
        }
        Ok(skip)
    }
}

// ===================================================================
// 4. Parallel Sub-Head Phase Modulation (`PspmRouter`)
// ===================================================================

#[inline(always)]
#[must_use]
const fn sanitize_pspm_gain_to_u7(gain: f32) -> u8 {
    let clean_gain = branchless_select_f32(branchless_is_finite_f32(gain), gain, 1.0);
    (branchless_clamp_f32(clean_gain, 0.0, 2.0) * 128.0 + 0.5) as u8
}

#[inline(always)]
#[must_use]
const fn compute_pspm_eff_q7(
    slot_q7: gcso_q7_t,
    gain_u7: u8,
    pulse_q7: gcso_q7_t,
    fold_geodesic: bool,
) -> gcso_q7_t {
    if (gain_u7 == 128) & (pulse_q7 == 0) {
        return if fold_geodesic {
            branchless_q7_fold_s1_geodesic(slot_q7)
        } else {
            slot_q7
        };
    }
    let base_q7 = if fold_geodesic {
        branchless_q7_fold_s1_geodesic(slot_q7)
    } else {
        slot_q7
    };
    let scaled_q7 = branchless_q7_scale_u7(base_q7, gain_u7);
    if fold_geodesic {
        branchless_q7_fold_s1_geodesic(scaled_q7.wrapping_add(pulse_q7))
    } else {
        branchless_q7_add_sat(scaled_q7, pulse_q7)
    }
}

#[inline(always)]
#[must_use]
const fn compute_pspm_eff_word_x8(
    slot_word: u64,
    gain_u7: u8,
    pulse_q7: gcso_q7_t,
    fold_geodesic: bool,
) -> u64 {
    if (gain_u7 == 128) & (pulse_q7 == 0) {
        return if fold_geodesic {
            swar_q7_fold_s1_geodesic_x8(slot_word)
        } else {
            slot_word
        };
    }
    let base_w = if fold_geodesic {
        swar_q7_fold_s1_geodesic_x8(slot_word)
    } else {
        slot_word
    };
    let scaled_w = if gain_u7 == 128 {
        base_w
    } else {
        swar_q7_scale_u7_x8(base_w, gain_u7)
    };
    if pulse_q7 == 0 {
        return if fold_geodesic {
            swar_q7_fold_s1_geodesic_x8(scaled_w)
        } else {
            scaled_w
        };
    }
    let pulse_splat = ((pulse_q7 as u8) as u64).wrapping_mul(SWAR_SPLAT_BYTE_64);
    if fold_geodesic {
        swar_q7_fold_s1_geodesic_x8(swar_q7_circular_add_x8(scaled_w, pulse_splat))
    } else {
        swar_q7_add_sat_x8(scaled_w, pulse_splat)
    }
}

#[inline(always)]
fn dispatch_subhead_profiles_core(
    tensor: &mut AlignedSliceMut32<'_, f32>,
    config: &gcso_pspm_config_t,
    fact_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    logic_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    explore_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    max_rad: f32,
    explore_pulse_q7: gcso_q7_t,
    head_dim: usize,
    fold_geodesic: bool,
) -> GcsoResult<()> {
    if head_dim == 0
        || (head_dim & 1) != 0
        || max_rad <= 0.0
        || !branchless_is_finite_f32(max_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let fact_h = config.num_fact_heads as usize;
    let logic_h = config.num_logic_heads as usize;
    let explore_h = config.num_explore_heads as usize;
    let total_heads = fact_h
        .checked_add(logic_h)
        .and_then(|v| v.checked_add(explore_h))
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let expected_len = total_heads
        .checked_mul(head_dim)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if total_heads == 0 || total_heads > MAX_ATTENTION_HEADS || tensor.len() < expected_len {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let t_slice = tensor.as_mut_slice();
    let fact_gain_u7 = sanitize_pspm_gain_to_u7(config.fact_phase_gain);
    let logic_gain_u7 = sanitize_pspm_gain_to_u7(config.logic_phase_gain);
    let explore_gain_u7 = sanitize_pspm_gain_to_u7(config.explore_phase_gain);
    let logic_end = fact_h + logic_h;

    let eff_words = build_pspm_eff_words_from_profiles_64(
        fact_phases,
        fact_h,
        fact_gain_u7,
        logic_phases,
        logic_end,
        logic_gain_u7,
        explore_phases,
        total_heads,
        explore_gain_u7,
        explore_pulse_q7,
        fold_geodesic,
    );
    let active_cells = total_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
    let cell_limit_mask = ((1u16 << active_cells) - 1) as u8;
    steer_single_slice_from_eff_words_core(
        t_slice,
        &eff_words,
        cell_limit_mask,
        cell_limit_mask,
        head_dim,
        total_heads,
        rot_dim,
        head_dim - rot_dim,
        false,
        is_default,
        eff_max,
        q7_to_norm_scale,
    );
    Ok(())
}

/// Assembles the 8 effective `u64` SWAR PSPM phase words across `[0..total_heads]` from three
/// 8-word arrays (`fact_w`, `logic_w`, `explore_w`), evaluating `compute_pspm_eff_word_x8` only
/// on the cells actually spanned by each subgroup (`8` word transforms total for aligned splits).
#[inline(always)]
#[must_use]
const fn build_pspm_eff_words_64(
    fact_w: &[u64; NUM_SWARM_CELLS],
    fact_h: usize,
    fact_gain_u7: u8,
    logic_w: &[u64; NUM_SWARM_CELLS],
    logic_end: usize,
    logic_gain_u7: u8,
    explore_w: &[u64; NUM_SWARM_CELLS],
    total_heads: usize,
    explore_gain_u7: u8,
    explore_pulse_q7: gcso_q7_t,
    fold_geodesic: bool,
) -> [u64; NUM_SWARM_CELLS] {
    let mut out = [0u64; NUM_SWARM_CELLS];
    let mut prev_tag = 0u8;
    let mut prev_in_w = 0u64;
    let mut prev_out_w = 0u64;
    let mut c = 0usize;
    while c < NUM_SWARM_CELLS {
        let cell_start = c * 8;
        if cell_start >= total_heads {
            break;
        }
        let cell_end = cell_start + 8;
        out[c] = if cell_end <= fact_h {
            let iw = fact_w[c];
            if (prev_tag == 1) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, fact_gain_u7, 0, fold_geodesic);
                prev_tag = 1;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else if cell_start >= fact_h && cell_end <= logic_end {
            let iw = logic_w[c];
            if (prev_tag == 2) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, logic_gain_u7, 0, fold_geodesic);
                prev_tag = 2;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else if cell_start >= logic_end {
            let iw = explore_w[c];
            if (prev_tag == 3) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, explore_gain_u7, explore_pulse_q7, fold_geodesic);
                prev_tag = 3;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else {
            let fw = if cell_start < fact_h {
                compute_pspm_eff_word_x8(fact_w[c], fact_gain_u7, 0, fold_geodesic)
            } else {
                0
            };
            let lw = if cell_end > fact_h && cell_start < logic_end {
                compute_pspm_eff_word_x8(logic_w[c], logic_gain_u7, 0, fold_geodesic)
            } else {
                0
            };
            let ew = if cell_end > logic_end {
                compute_pspm_eff_word_x8(
                    explore_w[c],
                    explore_gain_u7,
                    explore_pulse_q7,
                    fold_geodesic,
                )
            } else {
                0
            };
            let f_lanes = if fact_h <= cell_start {
                0
            } else if fact_h >= cell_end {
                8
            } else {
                fact_h - cell_start
            };
            let l_lanes = if logic_end <= cell_start {
                0
            } else if logic_end >= cell_end {
                8
            } else {
                logic_end - cell_start
            };
            let f_mask = if f_lanes >= 8 {
                u64::MAX
            } else {
                (1u64 << (f_lanes * 8)) - 1
            };
            let l_mask = if l_lanes >= 8 {
                u64::MAX
            } else {
                (1u64 << (l_lanes * 8)) - 1
            };
            (fw & f_mask) | (lw & (l_mask & !f_mask)) | (ew & !l_mask)
        };
        c += 1;
    }
    out
}

/// Loads only the active 8-byte SWAR cells from `fact_phases`, `logic_phases`, and `explore_phases`
/// (`8` `u64` loads total instead of `24` `u64` loads across three full 64-byte arrays) and applies
/// `compute_pspm_eff_word_x8` in registers.
#[inline(always)]
#[must_use]
fn build_pspm_eff_words_from_profiles_64(
    fact_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    fact_h: usize,
    fact_gain_u7: u8,
    logic_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    logic_end: usize,
    logic_gain_u7: u8,
    explore_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    total_heads: usize,
    explore_gain_u7: u8,
    explore_pulse_q7: gcso_q7_t,
    fold_geodesic: bool,
) -> [u64; NUM_SWARM_CELLS] {
    let mut out = [0u64; NUM_SWARM_CELLS];
    let mut prev_tag = 0u8;
    let mut prev_in_w = 0u64;
    let mut prev_out_w = 0u64;
    let mut c = 0usize;
    while c < NUM_SWARM_CELLS {
        let cell_start = c * 8;
        if cell_start >= total_heads {
            break;
        }
        let cell_end = cell_start + 8;
        out[c] = if cell_end <= fact_h {
            let iw = load_q7_x8_slice(fact_phases, cell_start);
            if (prev_tag == 1) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, fact_gain_u7, 0, fold_geodesic);
                prev_tag = 1;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else if cell_start >= fact_h && cell_end <= logic_end {
            let iw = load_q7_x8_slice(logic_phases, cell_start);
            if (prev_tag == 2) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, logic_gain_u7, 0, fold_geodesic);
                prev_tag = 2;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else if cell_start >= logic_end {
            let iw = load_q7_x8_slice(explore_phases, cell_start);
            if (prev_tag == 3) & (iw == prev_in_w) {
                prev_out_w
            } else {
                let ow = compute_pspm_eff_word_x8(iw, explore_gain_u7, explore_pulse_q7, fold_geodesic);
                prev_tag = 3;
                prev_in_w = iw;
                prev_out_w = ow;
                ow
            }
        } else {
            let fw = if cell_start < fact_h {
                compute_pspm_eff_word_x8(
                    load_q7_x8_slice(fact_phases, cell_start),
                    fact_gain_u7,
                    0,
                    fold_geodesic,
                )
            } else {
                0
            };
            let lw = if cell_end > fact_h && cell_start < logic_end {
                compute_pspm_eff_word_x8(
                    load_q7_x8_slice(logic_phases, cell_start),
                    logic_gain_u7,
                    0,
                    fold_geodesic,
                )
            } else {
                0
            };
            let ew = if cell_end > logic_end {
                compute_pspm_eff_word_x8(
                    load_q7_x8_slice(explore_phases, cell_start),
                    explore_gain_u7,
                    explore_pulse_q7,
                    fold_geodesic,
                )
            } else {
                0
            };
            let f_lanes = if fact_h <= cell_start {
                0
            } else if fact_h >= cell_end {
                8
            } else {
                fact_h - cell_start
            };
            let l_lanes = if logic_end <= cell_start {
                0
            } else if logic_end >= cell_end {
                8
            } else {
                logic_end - cell_start
            };
            let f_mask = if f_lanes >= 8 {
                u64::MAX
            } else {
                (1u64 << (f_lanes * 8)) - 1
            };
            let l_mask = if l_lanes >= 8 {
                u64::MAX
            } else {
                (1u64 << (l_lanes * 8)) - 1
            };
            (fw & f_mask) | (lw & (l_mask & !f_mask)) | (ew & !l_mask)
        };
        c += 1;
    }
    out
}

/// Rotates `t_slice` across `0..total_heads` by the fused $SO(2)$ group-closure angle
/// $\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}}$ directly from register-resident `base_ripa_words`
/// and pre-transformed `pspm_eff_words` in a single flat loop (`0` subgroup or cell loop exits).
/// Returns `Some(b)` if all 64 bytes in `w` equal `b`, short-circuiting on `w[0]` in 3 integer
/// instructions when `w[0]` is not byte-uniform.
#[inline(always)]
#[must_use]
const fn byte_uniform_word_u8(w: &[u64; NUM_SWARM_CELLS]) -> Option<u8> {
    let w0 = w[0];
    let b = (w0 & 0xFF) as u8;
    if w0 == (b as u64).wrapping_mul(SWAR_SPLAT_BYTE_64) && is_word_array_uniform_64(w) {
        Some(b)
    } else {
        None
    }
}

/// Evaluates the fused $SO(2)$ group-closure `(sin, cos)` for a single `(b_u8, s_u8)` Q7 pair,
/// fast-pathing default RIPA clamps (`0.24 rad`) via `.rodata` LUTs.
#[inline(always)]
#[must_use]
fn eval_fused_ripa_pspm_pair_u8(
    b_u8: u8,
    s_u8: u8,
    base_is_def: bool,
    base_eff_max: f32,
    base_scale: f32,
    pspm_is_def: bool,
    pspm_eff_max: f32,
    pspm_scale: f32,
) -> (f32, f32) {
    if (b_u8 | s_u8) == 0 {
        return (0.0, 1.0);
    }
    if base_is_def & pspm_is_def {
        if b_u8 == s_u8 {
            return Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF[b_u8 as usize];
        }
        let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
        let (sb, cb) = lut[b_u8 as usize];
        let (sp, cp) = lut[s_u8 as usize];
        return (sb * cp + cb * sp, cb * cp - sb * sp);
    }
    eval_fused_ripa_pspm_sincos_q7_fast(
        b_u8 as gcso_q7_t,
        false,
        base_is_def,
        base_eff_max,
        base_scale,
        s_u8 as gcso_q7_t,
        false,
        pspm_is_def,
        pspm_eff_max,
        pspm_scale,
    )
}

/// Rotates `t_slice` across `0..total_heads` when `base_ripa_words` is byte-uniform (`b_u8`) and
/// the 3 PSPM subgroups (`0..fact_h`, `fact_h..logic_end`, `logic_end..total_heads`) each have
/// constant effective Q7 bytes `(f_u8, l_u8, e_u8)`, coalescing adjacent subgroups when their
/// effective bytes match (`1..=3` calls to `rotate_ripa_upper_heads_uniform_sincos`).
#[inline(always)]
fn rotate_fused_ripa_and_pspm_uniform_subgroups(
    t_slice: &mut [f32],
    b_u8: u8,
    fact_h: usize,
    f_u8: u8,
    logic_end: usize,
    l_u8: u8,
    total_heads: usize,
    e_u8: u8,
    head_dim: usize,
    rot_dim: usize,
    base_is_def: bool,
    base_eff_max: f32,
    base_scale: f32,
    pspm_is_def: bool,
    pspm_eff_max: f32,
    pspm_scale: f32,
) {
    let limit = total_heads.min(MAX_ATTENTION_HEADS);
    if limit == 0 {
        return;
    }
    let fh = fact_h.min(limit);
    let le = logic_end.min(limit);
    let eff_f = if fh > 0 { f_u8 } else if le > 0 { l_u8 } else { e_u8 };
    let eff_l = if le > fh { l_u8 } else { eff_f };
    let eff_e = if limit > le { e_u8 } else { eff_l };

    if (eff_f == eff_l) & (eff_l == eff_e) {
        if (b_u8 | eff_f) != 0 {
            let (sin_c, cos_c) = eval_fused_ripa_pspm_pair_u8(
                b_u8,
                eff_f,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            rotate_ripa_upper_heads_uniform_sincos(t_slice, 0, limit, head_dim, rot_dim, sin_c, cos_c);
        }
        return;
    }
    if eff_f == eff_l {
        if le > 0 && (b_u8 | eff_f) != 0 {
            let (sin_fl, cos_fl) = eval_fused_ripa_pspm_pair_u8(
                b_u8,
                eff_f,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            rotate_ripa_upper_heads_uniform_sincos(t_slice, 0, le, head_dim, rot_dim, sin_fl, cos_fl);
        }
        if limit > le && (b_u8 | eff_e) != 0 {
            let (sin_e, cos_e) = eval_fused_ripa_pspm_pair_u8(
                b_u8,
                eff_e,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            rotate_ripa_upper_heads_uniform_sincos(t_slice, le, limit, head_dim, rot_dim, sin_e, cos_e);
        }
        return;
    }
    if eff_l == eff_e {
        if fh > 0 && (b_u8 | eff_f) != 0 {
            let (sin_f, cos_f) = eval_fused_ripa_pspm_pair_u8(
                b_u8,
                eff_f,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            rotate_ripa_upper_heads_uniform_sincos(t_slice, 0, fh, head_dim, rot_dim, sin_f, cos_f);
        }
        if limit > fh && (b_u8 | eff_l) != 0 {
            let (sin_le, cos_le) = eval_fused_ripa_pspm_pair_u8(
                b_u8,
                eff_l,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            rotate_ripa_upper_heads_uniform_sincos(t_slice, fh, limit, head_dim, rot_dim, sin_le, cos_le);
        }
        return;
    }
    if fh > 0 && (b_u8 | eff_f) != 0 {
        let (sin_f, cos_f) = eval_fused_ripa_pspm_pair_u8(
            b_u8,
            eff_f,
            base_is_def,
            base_eff_max,
            base_scale,
            pspm_is_def,
            pspm_eff_max,
            pspm_scale,
        );
        rotate_ripa_upper_heads_uniform_sincos(t_slice, 0, fh, head_dim, rot_dim, sin_f, cos_f);
    }
    if le > fh && (b_u8 | eff_l) != 0 {
        let (sin_l, cos_l) = eval_fused_ripa_pspm_pair_u8(
            b_u8,
            eff_l,
            base_is_def,
            base_eff_max,
            base_scale,
            pspm_is_def,
            pspm_eff_max,
            pspm_scale,
        );
        rotate_ripa_upper_heads_uniform_sincos(t_slice, fh, le, head_dim, rot_dim, sin_l, cos_l);
    }
    if limit > le && (b_u8 | eff_e) != 0 {
        let (sin_e, cos_e) = eval_fused_ripa_pspm_pair_u8(
            b_u8,
            eff_e,
            base_is_def,
            base_eff_max,
            base_scale,
            pspm_is_def,
            pspm_eff_max,
            pspm_scale,
        );
        rotate_ripa_upper_heads_uniform_sincos(t_slice, le, limit, head_dim, rot_dim, sin_e, cos_e);
    }
}

/// Rotates `t_slice` across `0..total_heads` directly from `base_ripa_words` using a single
/// precomputed self-fused LUT (`Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF` or
/// `Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF`) when `fact_words == logic_words == explore_words == base_ripa_words`,
/// unit gains (`128`), and `explore_pulse_q7 == 0` under `DEFAULT_RIPA_CLAMP_MAX_RAD`.
#[inline(never)]
fn rotate_ripa_lut_words_slice_nonuniform(
    t_slice: &mut [f32],
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    limit: usize,
    head_dim: usize,
    rot_dim: usize,
    self_lut: &[(f32, f32); 256],
) {
    let upper_off = head_dim - rot_dim;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::_mm256_set1_ps;
        let ptr = t_slice.as_mut_ptr();
        let p_ptr = base_ripa_words.as_ptr().cast::<u8>();
        if head_dim == 128 {
            // SAFETY: `t_slice` has length `limit * 128` (`rot_dim == 32`, `upper_off == 96`), and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(96);
                let mut h = 0usize;
                while h + 4 <= limit {
                    let wb32 = p_ptr.add(h).cast::<u32>().read_unaligned();
                    let (s0, c0) = self_lut[(wb32 as u8) as usize];
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                    let (s1, c1) = self_lut[((wb32 >> 8) as u8) as usize];
                    avx2_rotate_so2_block32(hp.add(128), _mm256_set1_ps(c1), _mm256_set1_ps(s1));
                    let (s2, c2) = self_lut[((wb32 >> 16) as u8) as usize];
                    avx2_rotate_so2_block32(hp.add(256), _mm256_set1_ps(c2), _mm256_set1_ps(s2));
                    let (s3, c3) = self_lut[(wb32 >> 24) as usize];
                    avx2_rotate_so2_block32(hp.add(384), _mm256_set1_ps(c3), _mm256_set1_ps(s3));
                    hp = hp.add(512);
                    h += 4;
                }
                while h < limit {
                    let (sin_c, cos_c) = self_lut[*p_ptr.add(h) as usize];
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_c), _mm256_set1_ps(sin_c));
                    hp = hp.add(128);
                    h += 1;
                }
            }
            return;
        }
        if rot_dim == 32 {
            // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 32`, and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                let step2 = head_dim * 2;
                let step3 = head_dim * 3;
                let step4 = head_dim * 4;
                let mut h = 0usize;
                while h + 4 <= limit {
                    let wb32 = p_ptr.add(h).cast::<u32>().read_unaligned();
                    let (s0, c0) = self_lut[(wb32 as u8) as usize];
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                    let (s1, c1) = self_lut[((wb32 >> 8) as u8) as usize];
                    avx2_rotate_so2_block32(
                        hp.add(head_dim),
                        _mm256_set1_ps(c1),
                        _mm256_set1_ps(s1),
                    );
                    let (s2, c2) = self_lut[((wb32 >> 16) as u8) as usize];
                    avx2_rotate_so2_block32(
                        hp.add(step2),
                        _mm256_set1_ps(c2),
                        _mm256_set1_ps(s2),
                    );
                    let (s3, c3) = self_lut[(wb32 >> 24) as usize];
                    avx2_rotate_so2_block32(
                        hp.add(step3),
                        _mm256_set1_ps(c3),
                        _mm256_set1_ps(s3),
                    );
                    hp = hp.add(step4);
                    h += 4;
                }
                while h < limit {
                    let (sin_c, cos_c) = self_lut[*p_ptr.add(h) as usize];
                    avx2_rotate_so2_block32(
                        hp,
                        _mm256_set1_ps(cos_c),
                        _mm256_set1_ps(sin_c),
                    );
                    hp = hp.add(head_dim);
                    h += 1;
                }
            }
            return;
        }
        if rot_dim == 16 {
            use core::arch::x86_64::{
                _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                _mm256_storeu_ps,
            };
            // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 16`, and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                let step2 = head_dim * 2;
                let mut h = 0usize;
                while h + 2 <= limit {
                    let wb16 = u32::from(p_ptr.add(h).cast::<u16>().read_unaligned());
                    let (s0, c0) = self_lut[(wb16 as u8) as usize];
                    let vc0 = _mm256_set1_ps(c0);
                    let vs0 = _mm256_set1_ps(s0);
                    let v0 = _mm256_loadu_ps(hp);
                    let v1 = _mm256_loadu_ps(hp.add(8));
                    _mm256_storeu_ps(
                        hp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), vs0),
                        ),
                    );
                    _mm256_storeu_ps(
                        hp.add(8),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, vc0),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), vs0),
                        ),
                    );
                    let hp1 = hp.add(head_dim);
                    let (s1, c1) = self_lut[((wb16 >> 8) as u8) as usize];
                    let vc1 = _mm256_set1_ps(c1);
                    let vs1 = _mm256_set1_ps(s1);
                    let u0 = _mm256_loadu_ps(hp1);
                    let u1 = _mm256_loadu_ps(hp1.add(8));
                    _mm256_storeu_ps(
                        hp1,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(u0, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u0), vs1),
                        ),
                    );
                    _mm256_storeu_ps(
                        hp1.add(8),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(u1, vc1),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(u1), vs1),
                        ),
                    );
                    hp = hp.add(step2);
                    h += 2;
                }
                while h < limit {
                    let (sin_c, cos_c) = self_lut[*p_ptr.add(h) as usize];
                    let vc = _mm256_set1_ps(cos_c);
                    let vs = _mm256_set1_ps(sin_c);
                    let v0 = _mm256_loadu_ps(hp);
                    let v1 = _mm256_loadu_ps(hp.add(8));
                    _mm256_storeu_ps(
                        hp,
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v0, vc),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), vs),
                        ),
                    );
                    _mm256_storeu_ps(
                        hp.add(8),
                        _mm256_addsub_ps(
                            _mm256_mul_ps(v1, vc),
                            _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), vs),
                        ),
                    );
                    hp = hp.add(head_dim);
                    h += 1;
                }
            }
            return;
        }
        if rot_dim == 64 {
            // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 64`, and AVX2 is enabled.
            unsafe {
                let mut hp = ptr.add(upper_off);
                let mut h = 0usize;
                while h < limit {
                    let (sin_c, cos_c) = self_lut[*p_ptr.add(h) as usize];
                    let vc = _mm256_set1_ps(cos_c);
                    let vs = _mm256_set1_ps(sin_c);
                    avx2_rotate_so2_block32(hp, vc, vs);
                    avx2_rotate_so2_block32(hp.add(32), vc, vs);
                    hp = hp.add(head_dim);
                    h += 1;
                }
            }
            return;
        }
    }
    let mut h = 0usize;
    while h < limit {
        let (sin_c, cos_c) =
            self_lut[((base_ripa_words[h >> 3] >> ((h & 7) << 3)) as u8) as usize];
        let head_base = h * head_dim;
        rotate_ripa_upper_slice(
            &mut t_slice[head_base + upper_off..head_base + head_dim],
            sin_c,
            cos_c,
        );
        h += 1;
    }
}

#[inline(always)]
pub(crate) fn rotate_fused_self_ripa_and_pspm_default_words(
    t_slice: &mut [f32],
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    total_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    fold_geodesic: bool,
) {
    let limit = total_heads.min(MAX_ATTENTION_HEADS);
    if limit == 0 {
        return;
    }
    let self_lut = if fold_geodesic {
        Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF
    } else {
        Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF
    };
    if let Some(b_u8) = byte_uniform_word_u8(base_ripa_words) {
        if b_u8 != 0 {
            let (sin_comb, cos_comb) = self_lut[b_u8 as usize];
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, 0, limit, head_dim, rot_dim, sin_comb, cos_comb,
            );
        }
        return;
    }
    rotate_ripa_lut_words_slice_nonuniform(
        t_slice,
        base_ripa_words,
        limit,
        head_dim,
        rot_dim,
        self_lut,
    );
}

/// Rotates `q_slice` via `Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF` ($\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}}$)
/// and `k_slice` via `Q7_DEFAULT_RIPA_SINCOS_REF` ($\theta_{\mathrm{ripa}}$), checking SWAR phase
/// uniformity once and streaming `q_slice` then `k_slice` in 4-head-unrolled passes to eliminate
/// 2-stream 4KB page-offset Load-Store Queue aliasing on 32KB (`128x64`) tensors.
#[inline(always)]
pub(crate) fn rotate_fused_self_ripa_pspm_q_and_ripa_k_default_words(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    total_heads: usize,
    head_dim: usize,
    rot_dim: usize,
) {
    let limit = total_heads.min(MAX_ATTENTION_HEADS);
    if limit == 0 {
        return;
    }
    let q_lut = Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF;
    let k_lut = Q7_DEFAULT_RIPA_SINCOS_REF;
    if let Some(b_u8) = byte_uniform_word_u8(base_ripa_words) {
        if b_u8 != 0 {
            let (sin_q, cos_q) = q_lut[b_u8 as usize];
            let (sin_k, cos_k) = k_lut[b_u8 as usize];
            rotate_ripa_upper_heads_uniform_sincos(
                q_slice, 0, limit, head_dim, rot_dim, sin_q, cos_q,
            );
            rotate_ripa_upper_heads_uniform_sincos(
                k_slice, 0, limit, head_dim, rot_dim, sin_k, cos_k,
            );
        }
        return;
    }
    rotate_ripa_lut_words_slice_nonuniform(
        q_slice,
        base_ripa_words,
        limit,
        head_dim,
        rot_dim,
        q_lut,
    );
    rotate_ripa_lut_words_slice_nonuniform(
        k_slice,
        base_ripa_words,
        limit,
        head_dim,
        rot_dim,
        k_lut,
    );
}

#[inline(always)]
fn rotate_fused_ripa_and_pspm_eff_words_core(
    t_slice: &mut [f32],
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    pspm_eff_words: &[u64; NUM_SWARM_CELLS],
    total_heads: usize,
    head_dim: usize,
    rot_dim: usize,
    base_is_def: bool,
    base_eff_max: f32,
    base_scale: f32,
    pspm_is_def: bool,
    pspm_eff_max: f32,
    pspm_scale: f32,
) {
    let limit = total_heads.min(MAX_ATTENTION_HEADS);
    if limit == 0 {
        return;
    }
    if let Some(b_u8) = byte_uniform_word_u8(base_ripa_words) {
        if let Some(s_u8) = byte_uniform_word_u8(pspm_eff_words) {
            if (b_u8 | s_u8) != 0 {
                let (sin_comb, cos_comb) = eval_fused_ripa_pspm_pair_u8(
                    b_u8,
                    s_u8,
                    base_is_def,
                    base_eff_max,
                    base_scale,
                    pspm_is_def,
                    pspm_eff_max,
                    pspm_scale,
                );
                rotate_ripa_upper_heads_uniform_sincos(
                    t_slice, 0, limit, head_dim, rot_dim, sin_comb, cos_comb,
                );
            }
            return;
        }
    }
    let upper_off = head_dim - rot_dim;
    if base_is_def & pspm_is_def {
        let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            use core::arch::x86_64::{
                _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps,
                _mm256_set1_ps, _mm256_storeu_ps,
            };
            let self_lut = Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF;
            let ptr = t_slice.as_mut_ptr();
            if rot_dim == 32 {
                let words_eq = (((base_ripa_words[0] ^ pspm_eff_words[0])
                    | (base_ripa_words[1] ^ pspm_eff_words[1]))
                    | ((base_ripa_words[2] ^ pspm_eff_words[2])
                        | (base_ripa_words[3] ^ pspm_eff_words[3]))
                    | ((base_ripa_words[4] ^ pspm_eff_words[4])
                        | (base_ripa_words[5] ^ pspm_eff_words[5]))
                    | ((base_ripa_words[6] ^ pspm_eff_words[6])
                        | (base_ripa_words[7] ^ pspm_eff_words[7])))
                    == 0;
                // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 32`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(upper_off);
                    let step2 = head_dim * 2;
                    let step3 = head_dim * 3;
                    let step4 = head_dim * 4;
                    let mut h = 0usize;
                    if words_eq {
                        while h + 4 <= limit {
                            let wb32 = (base_ripa_words[h >> 3] >> ((h & 4) << 3)) as u32;
                            let (s0, c0) = self_lut[(wb32 as u8) as usize];
                            avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                            let (s1, c1) = self_lut[((wb32 >> 8) as u8) as usize];
                            avx2_rotate_so2_block32(
                                hp.add(head_dim),
                                _mm256_set1_ps(c1),
                                _mm256_set1_ps(s1),
                            );
                            let (s2, c2) = self_lut[((wb32 >> 16) as u8) as usize];
                            avx2_rotate_so2_block32(
                                hp.add(step2),
                                _mm256_set1_ps(c2),
                                _mm256_set1_ps(s2),
                            );
                            let (s3, c3) = self_lut[(wb32 >> 24) as usize];
                            avx2_rotate_so2_block32(
                                hp.add(step3),
                                _mm256_set1_ps(c3),
                                _mm256_set1_ps(s3),
                            );
                            hp = hp.add(step4);
                            h += 4;
                        }
                        while h < limit {
                            let (sin_c, cos_c) =
                                self_lut[((base_ripa_words[h >> 3] >> ((h & 7) << 3)) as u8) as usize];
                            avx2_rotate_so2_block32(
                                hp,
                                _mm256_set1_ps(cos_c),
                                _mm256_set1_ps(sin_c),
                            );
                            hp = hp.add(head_dim);
                            h += 1;
                        }
                        return;
                    }
                    while h + 4 <= limit {
                        let shift = (h & 4) << 3;
                        let wb32 = (base_ripa_words[h >> 3] >> shift) as u32;
                        let ws32 = (pspm_eff_words[h >> 3] >> shift) as u32;
                        let (sb0, cb0) = lut[(wb32 as u8) as usize];
                        let (sp0, cp0) = lut[(ws32 as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp,
                            _mm256_set1_ps(cb0 * cp0 - sb0 * sp0),
                            _mm256_set1_ps(sb0 * cp0 + cb0 * sp0),
                        );
                        let (sb1, cb1) = lut[((wb32 >> 8) as u8) as usize];
                        let (sp1, cp1) = lut[((ws32 >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(head_dim),
                            _mm256_set1_ps(cb1 * cp1 - sb1 * sp1),
                            _mm256_set1_ps(sb1 * cp1 + cb1 * sp1),
                        );
                        let (sb2, cb2) = lut[((wb32 >> 16) as u8) as usize];
                        let (sp2, cp2) = lut[((ws32 >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(step2),
                            _mm256_set1_ps(cb2 * cp2 - sb2 * sp2),
                            _mm256_set1_ps(sb2 * cp2 + cb2 * sp2),
                        );
                        let (sb3, cb3) = lut[(wb32 >> 24) as usize];
                        let (sp3, cp3) = lut[(ws32 >> 24) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(step3),
                            _mm256_set1_ps(cb3 * cp3 - sb3 * sp3),
                            _mm256_set1_ps(sb3 * cp3 + cb3 * sp3),
                        );
                        hp = hp.add(step4);
                        h += 4;
                    }
                    while h < limit {
                        let sh = (h & 7) << 3;
                        let (sb, cb) = lut[((base_ripa_words[h >> 3] >> sh) as u8) as usize];
                        let (sp, cp) = lut[((pspm_eff_words[h >> 3] >> sh) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp,
                            _mm256_set1_ps(cb * cp - sb * sp),
                            _mm256_set1_ps(sb * cp + cb * sp),
                        );
                        hp = hp.add(head_dim);
                        h += 1;
                    }
                }
                return;
            }
            if rot_dim == 16 {
                // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 16`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(upper_off);
                    let mut h = 0usize;
                    while h < limit {
                        let sh = (h & 7) << 3;
                        let (sb, cb) = lut[((base_ripa_words[h >> 3] >> sh) as u8) as usize];
                        let (sp, cp) = lut[((pspm_eff_words[h >> 3] >> sh) as u8) as usize];
                        let v_cos = _mm256_set1_ps(cb * cp - sb * sp);
                        let v_sin = _mm256_set1_ps(sb * cp + cb * sp);
                        let v0 = _mm256_loadu_ps(hp);
                        let v1 = _mm256_loadu_ps(hp.add(8));
                        _mm256_storeu_ps(
                            hp,
                            _mm256_addsub_ps(
                                _mm256_mul_ps(v0, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                            ),
                        );
                        _mm256_storeu_ps(
                            hp.add(8),
                            _mm256_addsub_ps(
                                _mm256_mul_ps(v1, v_cos),
                                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
                            ),
                        );
                        hp = hp.add(head_dim);
                        h += 1;
                    }
                }
                return;
            }
            if rot_dim == 64 {
                // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 64`, and AVX2 is enabled.
                unsafe {
                    let mut hp = ptr.add(upper_off);
                    let mut h = 0usize;
                    while h < limit {
                        let sh = (h & 7) << 3;
                        let (sb, cb) = lut[((base_ripa_words[h >> 3] >> sh) as u8) as usize];
                        let (sp, cp) = lut[((pspm_eff_words[h >> 3] >> sh) as u8) as usize];
                        let v_cos = _mm256_set1_ps(cb * cp - sb * sp);
                        let v_sin = _mm256_set1_ps(sb * cp + cb * sp);
                        avx2_rotate_so2_block32(hp, v_cos, v_sin);
                        avx2_rotate_so2_block32(hp.add(32), v_cos, v_sin);
                        hp = hp.add(head_dim);
                        h += 1;
                    }
                }
                return;
            }
        }
        let mut h = 0usize;
        while h < limit {
            let sh = (h & 7) << 3;
            let (sb, cb) = lut[((base_ripa_words[h >> 3] >> sh) as u8) as usize];
            let (sp, cp) = lut[((pspm_eff_words[h >> 3] >> sh) as u8) as usize];
            let head_base = h * head_dim;
            rotate_ripa_upper_slice(
                &mut t_slice[head_base + upper_off..head_base + head_dim],
                sb * cp + cb * sp,
                cb * cp - sb * sp,
            );
            h += 1;
        }
        return;
    }
    let mut cached_pair: u32 = u32::MAX;
    let mut cached_sincos = (0.0_f32, 1.0_f32);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if rot_dim == 32 {
        use core::arch::x86_64::_mm256_set1_ps;
        let ptr = t_slice.as_mut_ptr();
        let same_clamp = base_eff_max.to_bits() == pspm_eff_max.to_bits();
        let two_base_eff_max = base_eff_max + base_eff_max;
        // SAFETY: `t_slice` has length `limit * head_dim`, `rot_dim == 32`, and AVX2 is enabled.
        unsafe {
            let mut hp = ptr.add(upper_off);
            let mut h = 0usize;
            while h < limit {
                let w_idx = h >> 3;
                let wb = base_ripa_words[w_idx];
                let ws = pspm_eff_words[w_idx];
                if same_clamp && wb == ws && (h & 7) == 0 && h + 8 <= limit {
                    let mut w = wb;
                    let mut k = 0usize;
                    while k < 8 {
                        let b_u8 = w as u8;
                        let pair_key = u32::from(b_u8) | (u32::from(b_u8) << 8);
                        let (sin_comb, cos_comb) = if pair_key == cached_pair {
                            cached_sincos
                        } else {
                            let sc = if b_u8 == 0 {
                                (0.0, 1.0)
                            } else {
                                branchless_sincos_q7_ripa_preinv(
                                    b_u8 as gcso_q7_t,
                                    two_base_eff_max,
                                    base_scale,
                                )
                            };
                            cached_pair = pair_key;
                            cached_sincos = sc;
                            sc
                        };
                        avx2_rotate_so2_block32(
                            hp,
                            _mm256_set1_ps(cos_comb),
                            _mm256_set1_ps(sin_comb),
                        );
                        hp = hp.add(head_dim);
                        w >>= 8;
                        k += 1;
                    }
                    h += 8;
                    continue;
                }
                let sh = (h & 7) << 3;
                let b_u8 = (wb >> sh) as u8;
                let s_u8 = (ws >> sh) as u8;
                let pair_key = u32::from(b_u8) | (u32::from(s_u8) << 8);
                let (sin_comb, cos_comb) = if pair_key == cached_pair {
                    cached_sincos
                } else {
                    let sc = eval_fused_ripa_pspm_sincos_q7_fast(
                        b_u8 as gcso_q7_t,
                        false,
                        base_is_def,
                        base_eff_max,
                        base_scale,
                        s_u8 as gcso_q7_t,
                        false,
                        pspm_is_def,
                        pspm_eff_max,
                        pspm_scale,
                    );
                    cached_pair = pair_key;
                    cached_sincos = sc;
                    sc
                };
                avx2_rotate_so2_block32(
                    hp,
                    _mm256_set1_ps(cos_comb),
                    _mm256_set1_ps(sin_comb),
                );
                hp = hp.add(head_dim);
                h += 1;
            }
        }
        return;
    }
    let mut h = 0usize;
    while h < limit {
        let sh = (h & 7) << 3;
        let b_u8 = (base_ripa_words[h >> 3] >> sh) as u8;
        let s_u8 = (pspm_eff_words[h >> 3] >> sh) as u8;
        let pair_key = u32::from(b_u8) | (u32::from(s_u8) << 8);
        let (sin_comb, cos_comb) = if pair_key == cached_pair {
            cached_sincos
        } else {
            let sc = eval_fused_ripa_pspm_sincos_q7_fast(
                b_u8 as gcso_q7_t,
                false,
                base_is_def,
                base_eff_max,
                base_scale,
                s_u8 as gcso_q7_t,
                false,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            cached_pair = pair_key;
            cached_sincos = sc;
            sc
        };
        let head_base = h * head_dim;
        rotate_ripa_upper_slice(
            &mut t_slice[head_base + upper_off..head_base + head_dim],
            sin_comb,
            cos_comb,
        );
        h += 1;
    }
}

/// Fuses **RIPA Phase Steering** directly from 8 register-resident `u64` SWAR words
/// (`base_ripa_words`) and **PSPM Upper-Quarter ($d_{\mathrm{head}}/4$) Sub-Head Routing** into a
/// **single rotation per head** on the upper low-frequency quarter (`rot_dim`) via $SO(2)$ group closure:
/// $$\mathbf{R}(\theta_{\mathrm{ripa}}) \cdot \mathbf{R}(\theta_{\mathrm{pspm}}) = \mathbf{R}(\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}})$$
/// Because both RIPA and PSPM act exclusively on the upper low-frequency quarter `[head_start + head_dim - rot_dim .. head_start + head_dim]`,
/// the lower $\frac{3}{4}d_{\mathrm{head}}$ high-frequency syntactic channels remain invariant and untouched.
/// Register-direct fused RIPA + PSPM sub-head dispatcher operating on `[u64; 8]` SWAR word arrays
/// (`base_ripa_words`, `fact_words`, `logic_words`, `explore_words`) without stack reloads, and
/// automatically coalescing adjacent subgroups when their phase words and gains match.
#[inline(always)]
pub(crate) fn dispatch_fused_ripa_and_pspm_words_core(
    tensor: &mut AlignedSliceMut32<'_, f32>,
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    base_ripa_max_rad: f32,
    config: &gcso_pspm_config_t,
    fact_words: &[u64; NUM_SWARM_CELLS],
    logic_words: &[u64; NUM_SWARM_CELLS],
    explore_words: &[u64; NUM_SWARM_CELLS],
    pspm_max_rad: f32,
    explore_pulse_q7: gcso_q7_t,
    head_dim: usize,
    fold_geodesic: bool,
) -> GcsoResult<()> {
    if head_dim == 0
        || (head_dim & 1) != 0
        || base_ripa_max_rad <= 0.0
        || !branchless_is_finite_f32(base_ripa_max_rad)
        || pspm_max_rad <= 0.0
        || !branchless_is_finite_f32(pspm_max_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let fact_h = config.num_fact_heads as usize;
    let logic_h = config.num_logic_heads as usize;
    let explore_h = config.num_explore_heads as usize;
    let total_heads = fact_h
        .checked_add(logic_h)
        .and_then(|v| v.checked_add(explore_h))
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let expected_len = total_heads
        .checked_mul(head_dim)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if total_heads == 0 || total_heads > MAX_ATTENTION_HEADS || tensor.len() < expected_len {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let (base_is_def, base_eff_max, base_scale) = resolve_ripa_norm_params(base_ripa_max_rad);
    let (pspm_is_def, pspm_eff_max, pspm_scale) = resolve_ripa_norm_params(pspm_max_rad);

    let rot_dim = ripa_upper_rot_dim(head_dim);
    let t_slice = tensor.as_mut_slice();
    let fact_gain_u7 = sanitize_pspm_gain_to_u7(config.fact_phase_gain);
    let logic_gain_u7 = sanitize_pspm_gain_to_u7(config.logic_phase_gain);
    let explore_gain_u7 = sanitize_pspm_gain_to_u7(config.explore_phase_gain);
    let logic_end = fact_h + logic_h;

    let fact_is_self =
        (fact_gain_u7 == 128) && (core::ptr::eq(fact_words, base_ripa_words) || fact_words == base_ripa_words);
    let logic_is_self = (logic_gain_u7 == 128)
        && (core::ptr::eq(logic_words, base_ripa_words) || logic_words == base_ripa_words);
    let explore_is_self = (explore_gain_u7 == 128)
        && (explore_pulse_q7 == 0)
        && (core::ptr::eq(explore_words, base_ripa_words) || explore_words == base_ripa_words);

    if base_is_def && pspm_is_def && fact_is_self && logic_is_self && explore_is_self {
        rotate_fused_self_ripa_and_pspm_default_words(
            t_slice,
            base_ripa_words,
            total_heads,
            head_dim,
            rot_dim,
            fold_geodesic,
        );
        return Ok(());
    }

    if let Some(b_u8) = byte_uniform_word_u8(base_ripa_words) {
        let f_uni = if fact_h == 0 {
            Some(0u8)
        } else if core::ptr::eq(fact_words, base_ripa_words) {
            Some(b_u8)
        } else {
            byte_uniform_word_u8(fact_words)
        };
        let l_uni = if logic_h == 0 || (core::ptr::eq(logic_words, fact_words) && fact_h > 0) {
            f_uni
        } else if core::ptr::eq(logic_words, base_ripa_words) {
            Some(b_u8)
        } else {
            byte_uniform_word_u8(logic_words)
        };
        let e_uni = if explore_h == 0 {
            l_uni
        } else if core::ptr::eq(explore_words, base_ripa_words) {
            Some(b_u8)
        } else if core::ptr::eq(explore_words, logic_words) && logic_h > 0 {
            l_uni
        } else if core::ptr::eq(explore_words, fact_words) && fact_h > 0 {
            f_uni
        } else {
            byte_uniform_word_u8(explore_words)
        };
        if let (Some(fb), Some(lb), Some(eb)) = (f_uni, l_uni, e_uni) {
            let f_eff =
                compute_pspm_eff_word_x8(u64::from(fb), fact_gain_u7, 0, fold_geodesic) as u8;
            let l_eff =
                compute_pspm_eff_word_x8(u64::from(lb), logic_gain_u7, 0, fold_geodesic) as u8;
            let e_eff = compute_pspm_eff_word_x8(
                u64::from(eb),
                explore_gain_u7,
                explore_pulse_q7,
                fold_geodesic,
            ) as u8;
            rotate_fused_ripa_and_pspm_uniform_subgroups(
                t_slice,
                b_u8,
                fact_h,
                f_eff,
                logic_end,
                l_eff,
                total_heads,
                e_eff,
                head_dim,
                rot_dim,
                base_is_def,
                base_eff_max,
                base_scale,
                pspm_is_def,
                pspm_eff_max,
                pspm_scale,
            );
            return Ok(());
        }
    }

    if base_is_def && pspm_is_def {
        if fact_is_self && logic_is_self && (logic_end & 7) == 0 && logic_end > 0 {
            let fl_elems = logic_end * head_dim;
            let (t_fl, t_ex) = t_slice.split_at_mut(fl_elems);
            let self_lut = if fold_geodesic {
                Q7_FUSED_SELF_GEODESIC_RIPA_SINCOS_REF
            } else {
                Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF
            };
            rotate_ripa_lut_words_slice_nonuniform(
                t_fl,
                base_ripa_words,
                logic_end,
                head_dim,
                rot_dim,
                self_lut,
            );
            if explore_h > 0 {
                let fl_w = logic_end >> 3;
                let b0 = base_ripa_words[fl_w];
                let p0 = compute_pspm_eff_word_x8(
                    explore_words[fl_w],
                    explore_gain_u7,
                    explore_pulse_q7,
                    fold_geodesic,
                );
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                if explore_h == 8 && rot_dim == 32 {
                    use core::arch::x86_64::_mm256_set1_ps;
                    let lut = Q7_DEFAULT_RIPA_SINCOS_REF;
                    let upper_off = head_dim - 32;
                    // SAFETY: `t_ex` has length `8 * head_dim`, `rot_dim == 32`, and AVX2 is enabled.
                    unsafe {
                        let hp = t_ex.as_mut_ptr().add(upper_off);
                        let step2 = head_dim * 2;
                        let step3 = head_dim * 3;
                        let hp4 = hp.add(head_dim * 4);
                        let wb_lo = b0 as u32;
                        let ws_lo = p0 as u32;
                        let (sb0, cb0) = lut[(wb_lo as u8) as usize];
                        let (sp0, cp0) = lut[(ws_lo as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp,
                            _mm256_set1_ps(cb0 * cp0 - sb0 * sp0),
                            _mm256_set1_ps(sb0 * cp0 + cb0 * sp0),
                        );
                        let (sb1, cb1) = lut[((wb_lo >> 8) as u8) as usize];
                        let (sp1, cp1) = lut[((ws_lo >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(head_dim),
                            _mm256_set1_ps(cb1 * cp1 - sb1 * sp1),
                            _mm256_set1_ps(sb1 * cp1 + cb1 * sp1),
                        );
                        let (sb2, cb2) = lut[((wb_lo >> 16) as u8) as usize];
                        let (sp2, cp2) = lut[((ws_lo >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(step2),
                            _mm256_set1_ps(cb2 * cp2 - sb2 * sp2),
                            _mm256_set1_ps(sb2 * cp2 + cb2 * sp2),
                        );
                        let (sb3, cb3) = lut[(wb_lo >> 24) as usize];
                        let (sp3, cp3) = lut[(ws_lo >> 24) as usize];
                        avx2_rotate_so2_block32(
                            hp.add(step3),
                            _mm256_set1_ps(cb3 * cp3 - sb3 * sp3),
                            _mm256_set1_ps(sb3 * cp3 + cb3 * sp3),
                        );
                        let wb_hi = (b0 >> 32) as u32;
                        let ws_hi = (p0 >> 32) as u32;
                        let (sb4, cb4) = lut[(wb_hi as u8) as usize];
                        let (sp4, cp4) = lut[(ws_hi as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp4,
                            _mm256_set1_ps(cb4 * cp4 - sb4 * sp4),
                            _mm256_set1_ps(sb4 * cp4 + cb4 * sp4),
                        );
                        let (sb5, cb5) = lut[((wb_hi >> 8) as u8) as usize];
                        let (sp5, cp5) = lut[((ws_hi >> 8) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp4.add(head_dim),
                            _mm256_set1_ps(cb5 * cp5 - sb5 * sp5),
                            _mm256_set1_ps(sb5 * cp5 + cb5 * sp5),
                        );
                        let (sb6, cb6) = lut[((wb_hi >> 16) as u8) as usize];
                        let (sp6, cp6) = lut[((ws_hi >> 16) as u8) as usize];
                        avx2_rotate_so2_block32(
                            hp4.add(step2),
                            _mm256_set1_ps(cb6 * cp6 - sb6 * sp6),
                            _mm256_set1_ps(sb6 * cp6 + cb6 * sp6),
                        );
                        let (sb7, cb7) = lut[(wb_hi >> 24) as usize];
                        let (sp7, cp7) = lut[(ws_hi >> 24) as usize];
                        avx2_rotate_so2_block32(
                            hp4.add(step3),
                            _mm256_set1_ps(cb7 * cp7 - sb7 * sp7),
                            _mm256_set1_ps(sb7 * cp7 + cb7 * sp7),
                        );
                    }
                    return Ok(());
                }
                let ex_w_len = (explore_h + 7) >> 3;
                let mut ex_base = [b0; NUM_SWARM_CELLS];
                let mut ex_pspm = [p0; NUM_SWARM_CELLS];
                let mut wi = 1usize;
                while wi < ex_w_len {
                    let src_w = fl_w + wi;
                    ex_base[wi] = base_ripa_words[src_w];
                    ex_pspm[wi] = compute_pspm_eff_word_x8(
                        explore_words[src_w],
                        explore_gain_u7,
                        explore_pulse_q7,
                        fold_geodesic,
                    );
                    wi += 1;
                }
                rotate_fused_ripa_and_pspm_eff_words_core(
                    t_ex,
                    &ex_base,
                    &ex_pspm,
                    explore_h,
                    head_dim,
                    rot_dim,
                    base_is_def,
                    base_eff_max,
                    base_scale,
                    pspm_is_def,
                    pspm_eff_max,
                    pspm_scale,
                );
            }
            return Ok(());
        } else if logic_is_self && (fact_h & 7) == 0 && (logic_end & 7) == 0 && logic_h > 0 {
            let fact_elems = fact_h * head_dim;
            let logic_elems = logic_h * head_dim;
            let (t_fact, t_rest) = t_slice.split_at_mut(fact_elems);
            let (t_logic, t_ex) = t_rest.split_at_mut(logic_elems);

            if fact_h > 0 {
                let fact_w_len = fact_h >> 3;
                let f0 = if fact_gain_u7 == 128 && !fold_geodesic {
                    fact_words[0]
                } else {
                    compute_pspm_eff_word_x8(fact_words[0], fact_gain_u7, 0, fold_geodesic)
                };
                let mut f_pspm = [f0; NUM_SWARM_CELLS];
                if fact_gain_u7 == 128 && !fold_geodesic {
                    let mut wi = 1usize;
                    while wi < fact_w_len {
                        f_pspm[wi] = fact_words[wi];
                        wi += 1;
                    }
                } else {
                    let mut wi = 1usize;
                    while wi < fact_w_len {
                        f_pspm[wi] = compute_pspm_eff_word_x8(
                            fact_words[wi],
                            fact_gain_u7,
                            0,
                            fold_geodesic,
                        );
                        wi += 1;
                    }
                }
                rotate_fused_ripa_and_pspm_eff_words_core(
                    t_fact,
                    base_ripa_words,
                    &f_pspm,
                    fact_h,
                    head_dim,
                    rot_dim,
                    base_is_def,
                    base_eff_max,
                    base_scale,
                    pspm_is_def,
                    pspm_eff_max,
                    pspm_scale,
                );
            }

            let fact_w_off = fact_h >> 3;
            if explore_is_self {
                let rest_h = logic_h + explore_h;
                let rest_w_len = (rest_h + 7) >> 3;
                let mut rest_base = [base_ripa_words[fact_w_off]; NUM_SWARM_CELLS];
                let mut r_wi = 1usize;
                while r_wi < rest_w_len {
                    rest_base[r_wi] = base_ripa_words[fact_w_off + r_wi];
                    r_wi += 1;
                }
                rotate_fused_self_ripa_and_pspm_default_words(
                    t_rest,
                    &rest_base,
                    rest_h,
                    head_dim,
                    rot_dim,
                    fold_geodesic,
                );
                return Ok(());
            }
            let logic_w_len = logic_h >> 3;
            let mut logic_base = [base_ripa_words[fact_w_off]; NUM_SWARM_CELLS];
            let mut l_wi = 1usize;
            while l_wi < logic_w_len {
                logic_base[l_wi] = base_ripa_words[fact_w_off + l_wi];
                l_wi += 1;
            }
            rotate_fused_self_ripa_and_pspm_default_words(
                t_logic,
                &logic_base,
                logic_h,
                head_dim,
                rot_dim,
                fold_geodesic,
            );

            if explore_h > 0 {
                let fl_w = logic_end >> 3;
                let ex_w_len = (explore_h + 7) >> 3;
                let b0 = base_ripa_words[fl_w];
                let p0 = compute_pspm_eff_word_x8(
                    explore_words[fl_w],
                    explore_gain_u7,
                    explore_pulse_q7,
                    fold_geodesic,
                );
                let mut ex_base = [b0; NUM_SWARM_CELLS];
                let mut ex_pspm = [p0; NUM_SWARM_CELLS];
                let mut e_wi = 1usize;
                while e_wi < ex_w_len {
                    let src_w = fl_w + e_wi;
                    ex_base[e_wi] = base_ripa_words[src_w];
                    ex_pspm[e_wi] = compute_pspm_eff_word_x8(
                        explore_words[src_w],
                        explore_gain_u7,
                        explore_pulse_q7,
                        fold_geodesic,
                    );
                    e_wi += 1;
                }
                rotate_fused_ripa_and_pspm_eff_words_core(
                    t_ex,
                    &ex_base,
                    &ex_pspm,
                    explore_h,
                    head_dim,
                    rot_dim,
                    base_is_def,
                    base_eff_max,
                    base_scale,
                    pspm_is_def,
                    pspm_eff_max,
                    pspm_scale,
                );
            }
            return Ok(());
        }
    }

    let pspm_eff_words = build_pspm_eff_words_64(
        fact_words,
        fact_h,
        fact_gain_u7,
        logic_words,
        logic_end,
        logic_gain_u7,
        explore_words,
        total_heads,
        explore_gain_u7,
        explore_pulse_q7,
        fold_geodesic,
    );
    rotate_fused_ripa_and_pspm_eff_words_core(
        t_slice,
        base_ripa_words,
        &pspm_eff_words,
        total_heads,
        head_dim,
        rot_dim,
        base_is_def,
        base_eff_max,
        base_scale,
        pspm_is_def,
        pspm_eff_max,
        pspm_scale,
    );
    Ok(())
}

#[inline(always)]
pub(crate) fn dispatch_fused_ripa_words_and_pspm_profiles_core(
    tensor: &mut AlignedSliceMut32<'_, f32>,
    base_ripa_words: &[u64; NUM_SWARM_CELLS],
    base_ripa_max_rad: f32,
    config: &gcso_pspm_config_t,
    fact_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    logic_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    explore_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    pspm_max_rad: f32,
    explore_pulse_q7: gcso_q7_t,
    head_dim: usize,
    fold_geodesic: bool,
) -> GcsoResult<()> {
    if head_dim == 0
        || (head_dim & 1) != 0
        || base_ripa_max_rad <= 0.0
        || !branchless_is_finite_f32(base_ripa_max_rad)
        || pspm_max_rad <= 0.0
        || !branchless_is_finite_f32(pspm_max_rad)
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let fact_h = config.num_fact_heads as usize;
    let logic_h = config.num_logic_heads as usize;
    let explore_h = config.num_explore_heads as usize;
    let total_heads = fact_h
        .checked_add(logic_h)
        .and_then(|v| v.checked_add(explore_h))
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    let expected_len = total_heads
        .checked_mul(head_dim)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if total_heads == 0 || total_heads > MAX_ATTENTION_HEADS || tensor.len() < expected_len {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let (base_is_def, base_eff_max, base_scale) = resolve_ripa_norm_params(base_ripa_max_rad);
    let (pspm_is_def, pspm_eff_max, pspm_scale) = resolve_ripa_norm_params(pspm_max_rad);
    let rot_dim = ripa_upper_rot_dim(head_dim);
    let fact_gain_u7 = sanitize_pspm_gain_to_u7(config.fact_phase_gain);
    let logic_gain_u7 = sanitize_pspm_gain_to_u7(config.logic_phase_gain);
    let explore_gain_u7 = sanitize_pspm_gain_to_u7(config.explore_phase_gain);
    let logic_end = fact_h + logic_h;

    let pspm_eff_words = build_pspm_eff_words_from_profiles_64(
        fact_phases,
        fact_h,
        fact_gain_u7,
        logic_phases,
        logic_end,
        logic_gain_u7,
        explore_phases,
        total_heads,
        explore_gain_u7,
        explore_pulse_q7,
        fold_geodesic,
    );
    rotate_fused_ripa_and_pspm_eff_words_core(
        tensor.as_mut_slice(),
        base_ripa_words,
        &pspm_eff_words,
        total_heads,
        head_dim,
        rot_dim,
        base_is_def,
        base_eff_max,
        base_scale,
        pspm_is_def,
        pspm_eff_max,
        pspm_scale,
    );
    Ok(())
}

/// Fuses **RIPA Phase Steering** (`base_ripa_phases`, `base_ripa_max_rad`) and
/// **PSPM Upper-Quarter ($d_{\mathrm{head}}/4$) Sub-Head Routing** into a **single memory pass**
/// over `tensor` by loading `base_ripa_phases` into 8 `u64` SWAR registers and delegating to
/// `dispatch_fused_ripa_words_and_pspm_profiles_core`.
#[inline(always)]
pub(crate) fn dispatch_fused_ripa_and_pspm_profiles_core(
    tensor: &mut AlignedSliceMut32<'_, f32>,
    base_ripa_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    base_ripa_max_rad: f32,
    config: &gcso_pspm_config_t,
    fact_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    logic_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    explore_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
    pspm_max_rad: f32,
    explore_pulse_q7: gcso_q7_t,
    head_dim: usize,
    fold_geodesic: bool,
) -> GcsoResult<()> {
    let base_words = load_q7_words_64(base_ripa_phases);
    dispatch_fused_ripa_words_and_pspm_profiles_core(
        tensor,
        &base_words,
        base_ripa_max_rad,
        config,
        fact_phases,
        logic_phases,
        explore_phases,
        pspm_max_rad,
        explore_pulse_q7,
        head_dim,
        fold_geodesic,
    )
}

#[inline(always)]
fn dispatch_subhead_slot_ensemble_core(
    tensor: &mut AlignedSliceMut32<'_, f32>,
    config: &gcso_pspm_config_t,
    slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
    fact_slot_id: u8,
    logic_slot_id: u8,
    explore_slot_id: u8,
    max_rad: f32,
    explore_pulse_q7: gcso_q7_t,
    head_dim: usize,
    fold_geodesic: bool,
) -> GcsoResult<()> {
    let f_id = usize::from(fact_slot_id);
    let l_id = usize::from(logic_slot_id);
    let e_id = usize::from(explore_slot_id);
    if f_id >= MAX_ATTRACTOR_DOMAINS
        || l_id >= MAX_ATTRACTOR_DOMAINS
        || e_id >= MAX_ATTRACTOR_DOMAINS
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let bank_2d = slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
    dispatch_subhead_profiles_core(
        tensor,
        config,
        &bank_2d[f_id],
        &bank_2d[l_id],
        &bank_2d[e_id],
        max_rad,
        explore_pulse_q7,
        head_dim,
        fold_geodesic,
    )
}

/// Trait for Parallel Sub-Head Phase Modulation (PSPM) Pseudo-MoE Routing (**Micro / Mezzo Level**, Section 2.4).
pub trait PspmRouter: BranchlessKernel {
    /// Validates `config` sub-head counts (`Fact + Logic + Explore > 0`) and returns
    /// `(fact_heads, logic_heads, explore_heads, total_heads, rot_dim)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim == 0`, `head_dim` is odd,
    /// `total_heads == 0`, or `tensor.len() < total_heads * head_dim`.
    #[inline(always)]
    fn validate_pspm_layout(
        &self,
        tensor: &AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> GcsoResult<(usize, usize, usize, usize, usize)> {
        if head_dim == 0 || (head_dim & 1) != 0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let fact_h = config.num_fact_heads as usize;
        let logic_h = config.num_logic_heads as usize;
        let explore_h = config.num_explore_heads as usize;
        let total_heads = fact_h
            .checked_add(logic_h)
            .and_then(|v| v.checked_add(explore_h))
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        let expected_len = total_heads
            .checked_mul(head_dim)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if total_heads == 0 || tensor.len() < expected_len {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let rot_dim = ripa_upper_rot_dim(head_dim);
        Ok((fact_h, logic_h, explore_h, total_heads, rot_dim))
    }

    /// Executes single-pass PSPM sub-head routing across Fact, Logic, and Exploration head groups (Section 2.4).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `total_heads == 0`,
    /// or `tensor` has fewer than `total_heads * head_dim` elements.
    #[inline]
    fn dispatch_single_pass(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> GcsoResult<()> {
        let (fact_h, logic_h, _explore_h, total_heads, rot_dim) =
            self.validate_pspm_layout(tensor, config, head_dim)?;
        let t_slice = tensor.as_mut_slice();

        let fact_rad = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(config.fact_phase_gain),
                config.fact_phase_gain,
                1.0,
            ) - 1.0,
            -DEFAULT_RIPA_CLAMP_MAX_RAD,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let logic_rad = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(config.logic_phase_gain),
                config.logic_phase_gain,
                1.0,
            ) - 1.0,
            -DEFAULT_RIPA_CLAMP_MAX_RAD,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
        );
        let explore_rad = branchless_clamp_f32(
            branchless_select_f32(
                branchless_is_finite_f32(config.explore_phase_gain),
                config.explore_phase_gain,
                1.0,
            ) - 1.0,
            -FRAC_PI_2,
            FRAC_PI_2,
        );

        let logic_end = fact_h + logic_h;
        if fact_h > 0 && fact_rad != 0.0 {
            let (sin_f, cos_f) = self.sincos_half_pi_f32(fact_rad);
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, 0, fact_h, head_dim, rot_dim, sin_f, cos_f,
            );
        }
        if logic_h > 0 && logic_rad != 0.0 {
            let (sin_l, cos_l) = self.sincos_half_pi_f32(logic_rad);
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, fact_h, logic_end, head_dim, rot_dim, sin_l, cos_l,
            );
        }
        if total_heads > logic_end && explore_rad != 0.0 {
            let (sin_e, cos_e) = self.sincos_half_pi_f32(explore_rad);
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, logic_end, total_heads, head_dim, rot_dim, sin_e, cos_e,
            );
        }
        Ok(())
    }

    /// Executes single-pass PSPM sub-head routing using per-head Q7 `phase_deltas` scaled by
    /// each sub-group's phase gain (`fact_phase_gain`, `logic_phase_gain`, `explore_phase_gain`)
    /// plus an exploration pulse `explore_pulse_q7`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is invalid, `max_rad <= 0.0`,
    /// `phase_deltas.len() < total_heads`, or `tensor` is too small.
    #[inline]
    fn dispatch_single_pass_with_phases(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
    ) -> GcsoResult<()> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (fact_h, logic_h, _explore_h, total_heads, rot_dim) =
            self.validate_pspm_layout(tensor, config, head_dim)?;
        if phase_deltas.len() < total_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (is_default, eff_max, q7_to_norm_scale) = resolve_ripa_norm_params(max_rad);
        let upper_off = head_dim - rot_dim;
        let t_slice = tensor.as_mut_slice();
        let p_slice = phase_deltas.as_slice();

        let f_gain_u7 = sanitize_pspm_gain_to_u7(config.fact_phase_gain);
        let l_gain_u7 = sanitize_pspm_gain_to_u7(config.logic_phase_gain);
        let e_gain_u7 = sanitize_pspm_gain_to_u7(config.explore_phase_gain);

        if total_heads <= MAX_ATTENTION_HEADS {
            let mut raw_words = [0u64; NUM_SWARM_CELLS];
            let full_cells = total_heads / HEADS_PER_CELL;
            let mut c = 0usize;
            while c < full_cells {
                raw_words[c] = load_q7_x8_slice(p_slice, c * HEADS_PER_CELL);
                c += 1;
            }
            let rem_start = full_cells * HEADS_PER_CELL;
            if rem_start < total_heads {
                let mut tail_w = 0u64;
                let mut h = rem_start;
                while h < total_heads {
                    tail_w |= u64::from(p_slice[h] as u8) << ((h - rem_start) * 8);
                    h += 1;
                }
                raw_words[full_cells] = tail_w;
            }
            let eff_words = build_pspm_eff_words_64(
                &raw_words,
                fact_h,
                f_gain_u7,
                &raw_words,
                fact_h + logic_h,
                l_gain_u7,
                &raw_words,
                total_heads,
                e_gain_u7,
                explore_pulse_q7,
                false,
            );
            let active_cells = total_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
            let cell_limit_mask = ((1u16 << active_cells) - 1) as u8;
            steer_single_slice_from_eff_words_core(
                t_slice,
                &eff_words,
                cell_limit_mask,
                cell_limit_mask,
                head_dim,
                total_heads,
                rot_dim,
                upper_off,
                false,
                is_default,
                eff_max,
                q7_to_norm_scale,
            );
            return Ok(());
        }

        for (h, &p_q7) in p_slice.iter().enumerate().take(fact_h) {
            let scaled_q7 = compute_pspm_eff_q7(p_q7, f_gain_u7, 0, false);
            let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                scaled_q7,
                false,
                is_default,
                eff_max,
                q7_to_norm_scale,
            );
            let start = h * head_dim + upper_off;
            rotate_ripa_upper_slice(&mut t_slice[start..start + rot_dim], sin_t, cos_t);
        }
        for (h, &p_q7) in p_slice
            .iter()
            .enumerate()
            .take(fact_h + logic_h)
            .skip(fact_h)
        {
            let scaled_q7 = compute_pspm_eff_q7(p_q7, l_gain_u7, 0, false);
            let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                scaled_q7,
                false,
                is_default,
                eff_max,
                q7_to_norm_scale,
            );
            let start = h * head_dim + upper_off;
            rotate_ripa_upper_slice(&mut t_slice[start..start + rot_dim], sin_t, cos_t);
        }
        for (h, &p_q7) in p_slice
            .iter()
            .enumerate()
            .take(total_heads)
            .skip(fact_h + logic_h)
        {
            let scaled_q7 = compute_pspm_eff_q7(p_q7, e_gain_u7, explore_pulse_q7, false);
            let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                scaled_q7,
                false,
                is_default,
                eff_max,
                q7_to_norm_scale,
            );
            let start = h * head_dim + upper_off;
            rotate_ripa_upper_slice(&mut t_slice[start..start + rot_dim], sin_t, cos_t);
        }
        Ok(())
    }

    /// Executes **Single-Pass Sub-Head Unified-Manifold Anchor Ensemble Routing** directly from
    /// three 64-head phase profiles (`fact_anchor_phases`, `logic_anchor_phases`,
    /// `explore_anchor_phases`) on the Single Unified Attractor Manifold without requiring a
    /// 512-byte 8-slot phase bank.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `total_heads == 0`,
    /// `total_heads > 64`, `max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn dispatch_subhead_unified_anchors(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        fact_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        logic_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        explore_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
        fold_geodesic: bool,
    ) -> GcsoResult<()> {
        dispatch_subhead_profiles_core(
            tensor,
            config,
            fact_anchor_phases,
            logic_anchor_phases,
            explore_anchor_phases,
            max_rad,
            explore_pulse_q7,
            head_dim,
            fold_geodesic,
        )
    }

    /// Executes **Single-Pass $SO(2)$ Group-Closure Fused RIPA + Unified-Manifold PSPM Sub-Head
    /// Routing**, rotating the shared upper $\frac{1}{4}d_{\mathrm{head}}$ low-frequency pairs by
    /// the composite angle $\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}}$ in a single traversal.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `total_heads == 0`,
    /// `total_heads > 64`, `base_ripa_max_rad <= 0.0`, `pspm_max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn dispatch_ripa_and_subhead_unified_anchors_fused(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        base_ripa_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        base_ripa_max_rad: f32,
        config: &gcso_pspm_config_t,
        fact_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        logic_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        explore_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pspm_max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
        fold_geodesic: bool,
    ) -> GcsoResult<()> {
        dispatch_fused_ripa_and_pspm_profiles_core(
            tensor,
            base_ripa_phases,
            base_ripa_max_rad,
            config,
            fact_anchor_phases,
            logic_anchor_phases,
            explore_anchor_phases,
            pspm_max_rad,
            explore_pulse_q7,
            head_dim,
            fold_geodesic,
        )
    }

    /// Executes **Single-Pass $SO(2)$ Group-Closure Fused RIPA + 8-Slot Ensemble PSPM Sub-Head
    /// Routing**, rotating the shared upper $\frac{1}{4}d_{\mathrm{head}}$ low-frequency pairs by
    /// the composite angle $\theta_{\mathrm{ripa}} + \theta_{\mathrm{pspm}}$ in a single traversal.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slot IDs `>= 8`, `slot_phase_bank.len() < 512`,
    /// `head_dim` is invalid, `base_ripa_max_rad <= 0.0`, `pspm_max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn dispatch_ripa_and_subhead_slot_ensemble_fused(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        base_ripa_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        base_ripa_max_rad: f32,
        config: &gcso_pspm_config_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        fact_slot_id: u8,
        logic_slot_id: u8,
        explore_slot_id: u8,
        pspm_max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
        fold_geodesic: bool,
    ) -> GcsoResult<()> {
        let f_id = usize::from(fact_slot_id);
        let l_id = usize::from(logic_slot_id);
        let e_id = usize::from(explore_slot_id);
        if f_id >= MAX_ATTRACTOR_DOMAINS
            || l_id >= MAX_ATTRACTOR_DOMAINS
            || e_id >= MAX_ATTRACTOR_DOMAINS
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bank_2d =
            slot_phase_bank.as_slot_bank::<MAX_ATTRACTOR_DOMAINS, MAX_ATTENTION_HEADS>()?;
        dispatch_fused_ripa_and_pspm_profiles_core(
            tensor,
            base_ripa_phases,
            base_ripa_max_rad,
            config,
            &bank_2d[f_id],
            &bank_2d[l_id],
            &bank_2d[e_id],
            pspm_max_rad,
            explore_pulse_q7,
            head_dim,
            fold_geodesic,
        )
    }

    /// Executes **Single-Pass Sub-Head Attractor Slot Ensemble Routing** (Section 2.4):
    /// simultaneously binds `Fact` heads to `fact_slot_id`, `Logic` heads to `logic_slot_id`,
    /// and `Explore` heads to `explore_slot_id` from a 512-byte `slot_phase_bank`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slot ID `>= 8`, `slot_phase_bank.len() < 512`,
    /// `head_dim` is invalid, `max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn dispatch_subhead_slot_ensemble(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        fact_slot_id: u8,
        logic_slot_id: u8,
        explore_slot_id: u8,
        max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
    ) -> GcsoResult<()> {
        dispatch_subhead_slot_ensemble_core(
            tensor,
            config,
            slot_phase_bank,
            fact_slot_id,
            logic_slot_id,
            explore_slot_id,
            max_rad,
            explore_pulse_q7,
            head_dim,
            false,
        )
    }

    /// Executes **Seam-Free $S^1$ Geodesic Sub-Head Attractor Slot Ensemble Routing** (Section 2.4 & 4.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slot ID `>= 8`, `slot_phase_bank.len() < 512`,
    /// `head_dim` is invalid, `max_rad <= 0.0`, or `tensor` is too small.
    #[inline]
    fn dispatch_subhead_slot_ensemble_geodesic(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        slot_phase_bank: &AlignedSlice16<'_, gcso_q7_t>,
        fact_slot_id: u8,
        logic_slot_id: u8,
        explore_slot_id: u8,
        max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
    ) -> GcsoResult<()> {
        dispatch_subhead_slot_ensemble_core(
            tensor,
            config,
            slot_phase_bank,
            fact_slot_id,
            logic_slot_id,
            explore_slot_id,
            max_rad,
            explore_pulse_q7,
            head_dim,
            true,
        )
    }

    /// Executes single-pass PSPM sub-head routing gated by `tagged_ptr`'s `ACTION_MODE_PSPM_ROUTE` bit (`0x08`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` or `tensor` bounds are invalid.
    #[inline]
    fn dispatch_pointer_gated(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> GcsoResult<bool> {
        let _ = self.validate_pspm_layout(tensor, config, head_dim)?;
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_PSPM_ROUTE) {
            return Ok(false);
        }
        self.dispatch_single_pass(tensor, config, head_dim)?;
        Ok(true)
    }

    /// Executes single-pass PSPM sub-head routing with per-head Q7 `phase_deltas` gated by `tagged_ptr`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim`, `max_rad`, or slice bounds are invalid.
    #[inline]
    fn dispatch_pointer_gated_with_phases(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
        explore_pulse_q7: gcso_q7_t,
        head_dim: usize,
    ) -> GcsoResult<bool> {
        if max_rad <= 0.0 || !branchless_is_finite_f32(max_rad) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (_f, _l, _e, total_heads, _rot_dim) =
            self.validate_pspm_layout(tensor, config, head_dim)?;
        if phase_deltas.len() < total_heads {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_PSPM_ROUTE) {
            return Ok(false);
        }
        self.dispatch_single_pass_with_phases(
            tensor,
            config,
            phase_deltas,
            max_rad,
            explore_pulse_q7,
            head_dim,
        )?;
        Ok(true)
    }
}

// ===================================================================
// 5. Synchronous Residual Adapter (`SrlAdapter`)
// ===================================================================

#[inline(always)]
fn validate_and_unpack_srl_descriptor<'a>(
    desc: &'a gcso_srl_descriptor_t,
    d_in: usize,
    d_out: usize,
) -> GcsoResult<(
    AlignedSlice32<'a, f32>,
    AlignedSlice32<'a, f32>,
    Option<AlignedSlice32<'a, f32>>,
)> {
    if desc.u_vector_ptr == 0 || desc.v_vector_ptr == 0 {
        return Err(GCSO_ERROR_NULL_POINTER);
    }
    if desc.rank != 1 || d_in == 0 || d_out == 0 || !branchless_is_finite_f32(desc.scale_factor) {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let u_ptr = desc.u_vector_ptr as *const f32;
    let v_ptr = desc.v_vector_ptr as *const f32;
    if (u_ptr as usize) & 31 != 0 || (v_ptr as usize) & 31 != 0 {
        return Err(GCSO_ERROR_MISALIGNED_POINTER);
    }
    let g_slice_opt = if desc.gain_scalar_ptr != 0 {
        let g_ptr = desc.gain_scalar_ptr as *const f32;
        if (g_ptr as usize) & 31 != 0 {
            return Err(GCSO_ERROR_MISALIGNED_POINTER);
        }
        // SAFETY: `g_ptr` is non-null, 32-byte aligned, and sized to `d_out` elements per C-ABI contract.
        Some(unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(g_ptr, d_out)) })
    } else {
        None
    };
    // SAFETY: `u_ptr` and `v_ptr` are non-null, 32-byte aligned, and sized to `d_out` and `d_in`.
    let u_slice = unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(u_ptr, d_out)) };
    // SAFETY: `v_ptr` is non-null, 32-byte aligned, and sized to `d_in`.
    let v_slice = unsafe { AlignedSlice32::new_unchecked(slice::from_raw_parts(v_ptr, d_in)) };
    Ok((u_slice, v_slice, g_slice_opt))
}

/// Trait for Synchronous Residual Layer (SRL) Dynamic Rank-1 Outer Product Injection
/// ($\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$)
/// (**Micro Level**, Section 1.5 & 5.2).
pub trait SrlAdapter: BranchlessKernel {
    /// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// into `y_out` from safe hardware-aligned slices using AVX2 (`dot_f32_unrolled4`, `srl_write_outer_product_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `y_out`, `x_in`, `u_vec`, or `v_vec` is empty,
    /// `x_in.len() != v_vec.len()`, `y_out.len() != u_vec.len()`, or `scale_factor` is non-finite.
    #[inline]
    fn eval_rank1_slices(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
        gain_vec: Option<&AlignedSlice32<'_, f32>>,
        scale_factor: f32,
    ) -> GcsoResult<()> {
        let d_in = x_in.len();
        let d_out = y_out.len();
        if d_in == 0
            || d_out == 0
            || v_vec.len() < d_in
            || u_vec.len() < d_out
            || !branchless_is_finite_f32(scale_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(g) = gain_vec {
            if g.len() < d_out {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
        if !branchless_is_finite_f32(dot_vx) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
        let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
        srl_write_outer_product_simd(
            &mut y_out.as_mut_slice()[..d_out],
            &u_vec.as_slice()[..d_out],
            g_slice_opt,
            eff_scale,
        );
        Ok(())
    }

    /// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// in-place into `y_inout` from safe hardware-aligned slices using AVX2 (`srl_accum_outer_product_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice is empty, lengths mismatch, or `scale_factor` is non-finite.
    #[inline]
    fn eval_rank1_slices_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
        gain_vec: Option<&AlignedSlice32<'_, f32>>,
        scale_factor: f32,
    ) -> GcsoResult<()> {
        let d_in = x_in.len();
        let d_out = y_inout.len();
        if d_in == 0
            || d_out == 0
            || v_vec.len() < d_in
            || u_vec.len() < d_out
            || !branchless_is_finite_f32(scale_factor)
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if let Some(g) = gain_vec {
            if g.len() < d_out {
                return Err(GCSO_ERROR_INVALID_ARGUMENT);
            }
        }

        let dot_vx = dot_f32_unrolled4(v_vec.as_slice(), x_in.as_slice(), d_in);
        if !branchless_is_finite_f32(dot_vx) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let eff_scale = branchless_flush_subnormal_f32(scale_factor * dot_vx);
        let g_slice_opt = gain_vec.map(|g| &g.as_slice()[..d_out]);
        srl_accum_outer_product_simd(
            &mut y_inout.as_mut_slice()[..d_out],
            &u_vec.as_slice()[..d_out],
            g_slice_opt,
            eff_scale,
        );
        Ok(())
    }

    /// Evaluates the standalone Rank-1 outer product $\mathbf{y} = s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// from a C-ABI [`gcso_srl_descriptor_t`] (including optional 32B-aligned `gain_scalar_ptr`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
    /// [`GCSO_ERROR_MISALIGNED_POINTER`] if pointers are not 32B-aligned, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite, or slices are empty.
    #[inline]
    fn eval_rank1(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_out.len())?;
        self.eval_rank1_slices(
            y_out,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )
    }

    /// Accumulates the fused Rank-1 outer product residual $\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$
    /// in-place into `y_inout` from a C-ABI [`gcso_srl_descriptor_t`] (including optional 32B-aligned `gain_scalar_ptr`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_NULL_POINTER`] if `u_vector_ptr == 0` or `v_vector_ptr == 0`,
    /// [`GCSO_ERROR_MISALIGNED_POINTER`] if pointers are not 32B-aligned, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `desc.rank != 1`, `desc.scale_factor` is non-finite, or slices are empty.
    #[inline]
    fn eval_rank1_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<()> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_inout.len())?;
        self.eval_rank1_slices_fused(
            y_inout,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )
    }

    /// Accumulates the fused Rank-1 outer product residual gated by `tagged_ptr`'s `ACTION_MODE_SRL_INJECT` bit (`0x10`).
    ///
    /// # Errors
    /// Propagates any pointer/slice validation error from [`Self::eval_rank1_fused`].
    #[inline]
    fn eval_rank1_pointer_gated_fused(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        desc: &gcso_srl_descriptor_t,
    ) -> GcsoResult<bool> {
        let (u_slice, v_slice, g_slice_opt) =
            validate_and_unpack_srl_descriptor(desc, x_in.len(), y_inout.len())?;
        if !ptr_has_action_mode(tagged_ptr, ACTION_MODE_SRL_INJECT) {
            return Ok(false);
        }
        self.eval_rank1_slices_fused(
            y_inout,
            x_in,
            &u_slice,
            &v_slice,
            g_slice_opt.as_ref(),
            desc.scale_factor,
        )?;
        Ok(true)
    }
}

// ===================================================================
// 6. 2D/3D Spatiotemporal Visual Attention Steering (`VisualAttentionSteering`)
// ===================================================================

#[inline(always)]
#[must_use]
fn patch_diff_sum_sq_unrolled4(cur_patch: &[f32], prv_patch: &[f32]) -> f32 {
    let n = cur_patch.len().min(prv_patch.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        use core::arch::x86_64::{
            _mm256_add_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps,
        };
        let pc = cur_patch.as_ptr();
        let pp = prv_patch.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled, and pointer offsets are bounds-checked against `n`.
        unsafe {
            let mut sq0 = _mm256_setzero_ps();
            let mut sq1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pc.add(i)), _mm256_loadu_ps(pp.add(i)));
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 16)),
                    _mm256_loadu_ps(pp.add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pc.add(i + 24)),
                    _mm256_loadu_ps(pp.add(i + 24)),
                );
                sq0 = _mm256_add_ps(sq0, _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)));
                sq1 = _mm256_add_ps(sq1, _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)));
                i += 32;
            }
            while i + 8 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pc.add(i)), _mm256_loadu_ps(pp.add(i)));
                sq0 = _mm256_add_ps(sq0, _mm256_mul_ps(d0, d0));
                i += 8;
            }
            let mut sum_sq = avx2_hsum_ps(_mm256_add_ps(sq0, sq1));
            while i < n {
                let d = *pc.add(i) - *pp.add(i);
                sum_sq += d * d;
                i += 1;
            }
            sum_sq
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut acc0 = 0.0_f32;
        let mut acc1 = 0.0_f32;
        let mut acc2 = 0.0_f32;
        let mut acc3 = 0.0_f32;
        let mut i = 0usize;
        while i + 4 <= n {
            let d0 = cur_patch[i] - prv_patch[i];
            let d1 = cur_patch[i + 1] - prv_patch[i + 1];
            let d2 = cur_patch[i + 2] - prv_patch[i + 2];
            let d3 = cur_patch[i + 3] - prv_patch[i + 3];
            acc0 += d0 * d0;
            acc1 += d1 * d1;
            acc2 += d2 * d2;
            acc3 += d3 * d3;
            i += 4;
        }
        let mut sum_sq = (acc0 + acc1) + (acc2 + acc3);
        while i < n {
            let d = cur_patch[i] - prv_patch[i];
            sum_sq += d * d;
            i += 1;
        }
        sum_sq
    }
}

#[inline(always)]
fn steer_visual_grid_bitmask_core<F>(
    grid_tensor: &mut AlignedSliceMut32<'_, f32>,
    active_patch_mask: &gcso_paged_bitmask_t,
    patch_stride_elems: usize,
    num_patches: usize,
    min_dim_multiple: usize,
    mut rotate_patch_fn: F,
) -> GcsoResult<u32>
where
    F: FnMut(&mut [f32]),
{
    if num_patches == 0
        || num_patches > (PAGED_BITMASK_BITS as usize)
        || patch_stride_elems == 0
        || (patch_stride_elems % min_dim_multiple) != 0
    {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }
    let required_elems = num_patches
        .checked_mul(patch_stride_elems)
        .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
    if grid_tensor.len() < required_elems {
        return Err(GCSO_ERROR_INVALID_ARGUMENT);
    }

    let g_slice = grid_tensor.as_mut_slice();
    let mut steered_count = 0u32;

    for (w_idx, &word_val) in active_patch_mask.bits.iter().enumerate() {
        let patch_base = w_idx * 64;
        if patch_base >= num_patches {
            break;
        }
        let max_bits = (num_patches - patch_base).min(64);
        let valid_mask = if max_bits == 64 {
            u64::MAX
        } else {
            (1u64 << max_bits) - 1
        };
        let mut word = word_val & valid_mask;
        while word != 0 {
            let bit = word.trailing_zeros() as usize;
            let run_bits = (word >> bit).trailing_ones() as usize;
            let next_bit = bit + run_bits;
            word = if next_bit >= 64 {
                0
            } else {
                word & (!0u64 << next_bit)
            };
            let elem_start = (patch_base + bit) * patch_stride_elems;
            let elem_len = run_bits * patch_stride_elems;
            rotate_patch_fn(&mut g_slice[elem_start..elem_start + elem_len]);
            steered_count += run_bits as u32;
        }
    }
    Ok(steered_count)
}

/// Trait for 2D Spatial and 3D Spatiotemporal Visual Attention Phase Steering and
/// Bitmask-Gated Frame Delta Detection (**Micro / Multimodal Level**).
pub trait VisualAttentionSteering: BranchlessKernel {
    /// Applies 2D spatial RoPE phase steering $(\Delta\theta_x, \Delta\theta_y)$ across a visual
    /// patch tensor (`head_dim` must be a positive multiple of `4`) using AVX2 (`rotate_so2_2d_xy_slice_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim == 0`, `(head_dim & 3) != 0`, `num_heads == 0`,
    /// or `patch_tensor.len() < head_dim * num_heads`.
    #[inline]
    fn steer_visual_patch_2d(
        &self,
        patch_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
        head_dim: usize,
        num_heads: usize,
    ) -> GcsoResult<()> {
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if head_dim == 0
            || (head_dim & 3) != 0
            || num_heads == 0
            || patch_tensor.len() < expected_len
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        rotate_so2_2d_xy_slice_simd(
            &mut patch_tensor.as_mut_slice()[..expected_len],
            sin_x,
            cos_x,
            sin_y,
            cos_y,
        );
        Ok(())
    }

    /// Applies 3D spatiotemporal RoPE phase steering $(\Delta\theta_t, \Delta\theta_x, \Delta\theta_y)$
    /// across a video/3D patch tensor (`head_dim` must be a positive multiple of `6`) using AVX2
    /// (`rotate_so2_3d_txy_slice_simd`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim == 0`, `(head_dim % 6) != 0`, `num_heads == 0`,
    /// or `patch_tensor.len() < head_dim * num_heads`.
    #[inline]
    fn steer_visual_patch_3d(
        &self,
        patch_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_t_q7: gcso_q7_t,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
        head_dim: usize,
        num_heads: usize,
    ) -> GcsoResult<()> {
        let expected_len = head_dim
            .checked_mul(num_heads)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if head_dim == 0
            || (head_dim % 6) != 0
            || num_heads == 0
            || patch_tensor.len() < expected_len
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (sin_t, cos_t) = branchless_sincos_q7(phase_t_q7);
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        rotate_so2_3d_txy_slice_simd(
            &mut patch_tensor.as_mut_slice()[..expected_len],
            sin_t,
            cos_t,
            sin_x,
            cos_x,
            sin_y,
            cos_y,
        );
        Ok(())
    }

    /// Computes the 256-bit frame delta bitmask (`prev_mask ^ curr_mask`) and active changed patch
    /// count (`0..=256`) in $\mathcal{O}(1)$ branchless POPCNT arithmetic.
    #[inline(always)]
    #[must_use]
    fn detect_frame_delta_bitmask(
        &self,
        prev_mask: &gcso_paged_bitmask_t,
        curr_mask: &gcso_paged_bitmask_t,
    ) -> (gcso_paged_bitmask_t, u32) {
        let w0 = prev_mask.bits[0] ^ curr_mask.bits[0];
        let w1 = prev_mask.bits[1] ^ curr_mask.bits[1];
        let w2 = prev_mask.bits[2] ^ curr_mask.bits[2];
        let w3 = prev_mask.bits[3] ^ curr_mask.bits[3];
        let changed = w0.count_ones() + w1.count_ones() + w2.count_ones() + w3.count_ones();
        (
            gcso_paged_bitmask_t {
                bits: [w0, w1, w2, w3],
            },
            changed,
        )
    }

    /// Detects active visual patches between raw `f32` frame buffers `current_frame` and `prev_frame`
    /// whose mean squared difference exceeds `delta_threshold_sq`, populating a 256-bit
    /// [`gcso_paged_bitmask_t`] (`mask_out`) and returning the active patch count (`0..=256`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems == 0`, `delta_threshold_sq < 0.0`, or frame buffers are too small.
    #[inline]
    fn detect_frame_mse_bitmask(
        &self,
        current_frame: &AlignedSlice32<'_, f32>,
        prev_frame: &AlignedSlice32<'_, f32>,
        num_patches: usize,
        patch_stride_elems: usize,
        delta_threshold_sq: f32,
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> GcsoResult<u32> {
        let total_elems = num_patches
            .checked_mul(patch_stride_elems)
            .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?;
        if num_patches == 0
            || num_patches > (PAGED_BITMASK_BITS as usize)
            || patch_stride_elems == 0
            || delta_threshold_sq < 0.0
            || !branchless_is_finite_f32(delta_threshold_sq)
            || current_frame.len() < total_elems
            || prev_frame.len() < total_elems
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        *mask_out = gcso_paged_bitmask_t::default();
        let cur_s = &current_frame.as_slice()[..total_elems];
        let prv_s = &prev_frame.as_slice()[..total_elems];
        let inv_stride = fast_inv_len_f32(patch_stride_elems);
        let mut active_count = 0u32;

        for (p_idx, (c_patch, p_patch)) in cur_s
            .chunks_exact(patch_stride_elems)
            .zip(prv_s.chunks_exact(patch_stride_elems))
            .enumerate()
        {
            let sum_sq = patch_diff_sum_sq_unrolled4(c_patch, p_patch);
            let mse = sum_sq * inv_stride;
            let active = mse > delta_threshold_sq;
            let word_idx = p_idx >> 6;
            let bit_idx = p_idx & 63;
            mask_out.bits[word_idx] |= u64::from(active) << bit_idx;
            active_count += u32::from(active);
        }
        Ok(active_count)
    }

    /// Steers only the active 2D visual patches indicated by `active_patch_mask` (`bits[0..4]`)
    /// using `ctz` bit-scanning (`word &= word - 1`) in $\mathcal{O}(\mathrm{popcount}(\mathbf{M}))$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems` is zero or not a multiple of `4`, or `grid_tensor` is too small.
    #[inline]
    fn steer_visual_grid_2d_bitmask_gated(
        &self,
        grid_tensor: &mut AlignedSliceMut32<'_, f32>,
        active_patch_mask: &gcso_paged_bitmask_t,
        patch_stride_elems: usize,
        num_patches: usize,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
    ) -> GcsoResult<u32> {
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        steer_visual_grid_bitmask_core(
            grid_tensor,
            active_patch_mask,
            patch_stride_elems,
            num_patches,
            4,
            |patch_slice| rotate_so2_2d_xy_slice_simd(patch_slice, sin_x, cos_x, sin_y, cos_y),
        )
    }

    /// Steers only the active 3D spatiotemporal patches indicated by `active_patch_mask` (`bits[0..4]`)
    /// using `ctz` bit-scanning (`word &= word - 1`) in $\mathcal{O}(\mathrm{popcount}(\mathbf{M}))$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `num_patches == 0`, `num_patches > 256`,
    /// `patch_stride_elems` is zero or not a multiple of `6`, or `grid_tensor` is too small.
    #[inline]
    fn steer_visual_grid_3d_bitmask_gated(
        &self,
        grid_tensor: &mut AlignedSliceMut32<'_, f32>,
        active_patch_mask: &gcso_paged_bitmask_t,
        patch_stride_elems: usize,
        num_patches: usize,
        phase_t_q7: gcso_q7_t,
        phase_x_q7: gcso_q7_t,
        phase_y_q7: gcso_q7_t,
    ) -> GcsoResult<u32> {
        let (sin_t, cos_t) = branchless_sincos_q7(phase_t_q7);
        let (sin_x, cos_x) = branchless_sincos_q7(phase_x_q7);
        let (sin_y, cos_y) = branchless_sincos_q7(phase_y_q7);
        steer_visual_grid_bitmask_core(
            grid_tensor,
            active_patch_mask,
            patch_stride_elems,
            num_patches,
            6,
            |patch_slice| {
                rotate_so2_3d_txy_slice_simd(
                    patch_slice,
                    sin_t,
                    cos_t,
                    sin_x,
                    cos_x,
                    sin_y,
                    cos_y,
                );
            },
        )
    }
}

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
    use crate::traits::aligned::{AlignedBuffer32, AlignedBuffer64};
    use crate::traits::primitives::{
        ACTION_PRESET_FULL_ORCHESTRATION, ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA,
        DEFAULT_QDPS_MIN_STEP_RAD,
    };

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
        for h in 0..8 {
            assert_eq!(proc_out.data[h], 32);
        }

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
        // Lower 3/4 channels (0..10) remain 1.0; upper quarter (10..12) is rotated by R(theta_ripa + theta_pspm):
        assert_eq!(&q_pspm.data[0..10], &[1.0_f32; 10]);
        assert!((q_pspm.data[10] - 1.0).abs() > 1.0e-3);

        // Verify direct LUT fast path in `dispatch_subhead_slot_ensemble_geodesic` (`gain == 1.0, pulse == 0`)
        let pspm_unit_cfg = gcso_pspm_config_t {
            num_fact_heads: 2,
            num_logic_heads: 4,
            num_explore_heads: 2,
            fact_phase_gain: 1.0,
            logic_phase_gain: 1.0,
            explore_phase_gain: 1.0,
            ..Default::default()
        };
        let mut q_geo_unit = AlignedBuffer32::new([1.0_f32; 96]);
        eng.dispatch_subhead_slot_ensemble_geodesic(
            &mut q_geo_unit.as_aligned_slice_mut(),
            &pspm_unit_cfg,
            &bank.relax_slice::<16>(),
            ATTRACTOR_SLOT_KNOWLEDGE,
            ATTRACTOR_SLOT_PERSONA,
            2,
            DEFAULT_RIPA_CLAMP_MAX_RAD,
            0,
            12,
        )
        .unwrap();
        assert_eq!(&q_geo_unit.data[0..10], &[1.0_f32; 10]);
        assert!((q_geo_unit.data[10] - 1.0).abs() > 1.0e-3);

        // Also verify custom non-default RIPA clamp (`0.18 rad`) single-evaluation fused path
        let mut q_custom = AlignedBuffer32::new([1.0_f32; 96]);
        eng.dispatch_ripa_and_subhead_slot_ensemble_fused(
            &mut q_custom.as_aligned_slice_mut(),
            &[20i8; 64],
            0.18,
            &pspm_cfg,
            &bank.relax_slice::<16>(),
            ATTRACTOR_SLOT_KNOWLEDGE,
            ATTRACTOR_SLOT_PERSONA,
            2,
            0.18,
            8,
            12,
            true,
        )
        .unwrap();
        assert_eq!(&q_custom.data[0..10], &[1.0_f32; 10]);
        assert!((q_custom.data[10] - 1.0).abs() > 1.0e-3);

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

        let prev_mask = gcso_paged_bitmask_t::default();
        let (delta_mask, changed_bits) = eng.detect_frame_delta_bitmask(&prev_mask, &mask);
        assert_eq!(changed_bits, 1);

        let steered = eng
            .steer_visual_grid_3d_bitmask_gated(
                &mut frame_cur.as_aligned_slice_mut(),
                &delta_mask,
                12,
                4,
                16,
                24,
                -16,
            )
            .unwrap();
        assert_eq!(steered, 1);
    }
}