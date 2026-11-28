// name: src/core/src/foundation/rope/kernels/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core SWAR/AVX2 RoPE rotation, `PhaseClamp`, SLERP, Procrustes, and Bayesian attention helpers.

use super::*;
use crate::foundation::simd_stats::atan2_to_q7_word_x8_simd;

/// Computes the `PhaseClamp` / sub-head router upper low-frequency rotation dimension
/// `rot_dim = ((head_dim / 4) & !1).max(2)` (even integer in `2..=head_dim`).
#[inline(always)]
#[must_use]
pub const fn phase_clamp_upper_rot_dim(head_dim: usize) -> usize {
    let quarter_even = (head_dim >> 2) & !1usize;
    let base_rot = if quarter_even >= 2 { quarter_even } else { 2 };
    if base_rot <= head_dim {
        base_rot
    } else {
        head_dim
    }
}

/// Computes the **Swarm- & Blanket-Driven Autonomous Frequency Aperture** (`rot_dim`, an even
/// integer in `2..=head_dim`) without manual hyperparameters:
/// - **Narrow Fact Aperture ($\frac{1}{8}d_{\mathrm{head}}$)**: Selected when `provenance_bits`
///   carries Objective Fact grounding (`CAUSAL_AXIS_OBJECTIVE_FACT`, bit `1`) without contradiction
///   or persona override, or when Toroidal $k$-Means concentration is very narrow ($\kappa_c \ge 6.0$),
///   preserving high- and mid-frequency syntactic/lexical RoPE channels.
/// - **Baseline `PhaseClamp` Aperture ($\frac{1}{4}d_{\mathrm{head}}$)**: Standard upper-quarter rotation
///   (`phase_clamp_upper_rot_dim(head_dim)`).
/// - **Expanded Persona / High-Quorum Aperture ($\frac{3}{8}d_{\mathrm{head}}$)**: Selected when
///   Subjective Persona grounding (`CAUSAL_AXIS_SUBJECTIVE_PERSONA`, bit `2`) resonates with high
///   Micro-Swarm quorum (`quorum_mask.count_ones() >= 6`) and moderate concentration, allowing
///   persona voice and cadence to steer mid-frequency syntactic phrasing.
/// - **Wide Repulsion / Bifurcation Escape Aperture ($\frac{1}{2}d_{\mathrm{head}}$)**: Selected
///   when Contradiction Guard repulsion (`CAUSAL_AXIS_CONTRADICTION_GUARD`, bit `3`) or an `EntropyController`
///   entropy surge ($\tilde{H} \ge 0.55$) occurs, rotating the upper half of RoPE frequencies to
///   escape spurious local minima ($H^1(K; \mathcal{F}) \neq 0$).
#[inline(always)]
#[must_use]
pub const fn resolve_autonomous_swarm_rot_dim(
    head_dim: usize,
    provenance_bits: u32,
    quorum_mask: u8,
    moving_z_entropy: f32,
    von_mises_kappa: f32,
) -> usize {
    let safe_dim = head_dim & !1usize;
    if safe_dim <= 2 {
        return if head_dim >= 2 { 2 } else { head_dim };
    }
    let eighth_even = if ((safe_dim >> 3) & !1usize) >= 2 {
        (safe_dim >> 3) & !1usize
    } else {
        2
    };
    let quarter_even = if ((safe_dim >> 2) & !1usize) >= 2 {
        (safe_dim >> 2) & !1usize
    } else {
        2
    };
    let three_eighths_raw = ((safe_dim * 3) >> 3) & !1usize;
    let three_eighths_even = if three_eighths_raw >= quarter_even {
        three_eighths_raw
    } else {
        quarter_even
    };
    let half_raw = (safe_dim >> 1) & !1usize;
    let half_even = if half_raw >= quarter_even {
        half_raw
    } else {
        quarter_even
    };

    let has_contradiction = (provenance_bits & 0x08) != 0;
    let has_persona = (provenance_bits & 0x04) != 0;
    let has_fact = (provenance_bits & 0x02) != 0;
    let entropy_surge = branchless_is_finite_f32(moving_z_entropy) && moving_z_entropy >= 0.55;
    let high_kappa = branchless_is_finite_f32(von_mises_kappa) && von_mises_kappa >= 6.0;
    let quorum_cells = quorum_mask.count_ones();

    let raw_rot = if has_contradiction || entropy_surge {
        half_even
    } else if has_persona && !has_fact && quorum_cells >= 6 && !high_kappa {
        three_eighths_even
    } else if (has_fact && !has_persona) || high_kappa {
        eighth_even
    } else {
        quarter_even
    };
    if raw_rot <= safe_dim {
        raw_rot
    } else {
        safe_dim
    }
}

#[inline(always)]
#[must_use]
pub(crate) const fn ptr_has_action_mode(tagged_ptr: TaggedPtr, mode_mask: u8) -> bool {
    (unpack_action_mode_from_tag(tagged_ptr.unpack_tag()) & mode_mask) == mode_mask
}

/// Resolves `(is_default_phase_clamp, effective_max_rad, q7_to_norm_scale)` for `PhaseClamp` steering.
#[inline(always)]
#[must_use]
pub(crate) const fn resolve_phase_clamp_norm_params(max_rad: f32) -> (bool, f32, f32) {
    let is_default = max_rad.to_bits() == DEFAULT_PHASE_CLAMP_MAX_RAD.to_bits();
    let eff_max = branchless_min_f32(max_rad, FRAC_PI_2);
    let q7_to_norm_scale = if is_default {
        DEFAULT_PHASE_CLAMP_Q7_TO_NORM_SCALE
    } else {
        Q7_PHASE_SCALE_RAD * branchless_inv_pos_f32(eff_max)
    };
    (is_default, eff_max, q7_to_norm_scale)
}

#[inline(always)]
#[must_use]
pub(crate) const fn eval_phase_clamp_sincos_with_params(
    raw_q7: PhaseQ7,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) -> (f32, f32) {
    if fold_geodesic {
        eval_geodesic_phase_clamp_sincos_q7_fast(raw_q7, is_default, eff_max, q7_to_norm_scale)
    } else {
        eval_phase_clamp_sincos_q7_fast(raw_q7, is_default, eff_max, q7_to_norm_scale)
    }
}

/// 4-argument AVX2 leaf kernel (`ptr`, `p_ptr`, `head_dim`, `num_heads`) that rotates
/// `ptr[0..num_heads * head_dim]` in-place using pointer arithmetic to minimize register pressure
/// and avoid inner-loop slice bounds checks across common head dimensions (`128`, `256`, and `512`).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
pub(super) unsafe fn apply_phase_steering_nonuniform_avx2_ptr(
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
            // Unroll 1 head (`4x avx2_rotate_so2_block32`) per iteration to keep the inner loop
            // body compact for instruction-cache / decoded-uop-cache residency.
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
            // Single flat loop over `num_heads * 2` 128-float blocks (`blk >> 1` phase index),
            // keeping the inner loop body compact for instruction-cache / decoded-uop-cache residency.
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
pub(super) fn apply_phase_steering_nonuniform_slice_core(
    t_slice: &mut [f32],
    p_slice: &[PhaseQ7],
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

/// Rotates `t_slice` across `num_heads` heads without branching on individual phase magnitudes,
/// coalescing uniform phase profiles (`is_phase_slice_uniform`) into a single-broadcast AVX2
/// streaming pass when all active heads share the same Q0.7 angle.
#[inline(never)]
pub(super) fn apply_phase_steering_slice_core(
    t_slice: &mut [f32],
    p_slice: &[PhaseQ7],
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
                let (sin_t, cos_t) =
                    Q7_SINCOS_LUT_256_REF[(*p_slice.get_unchecked(0) as u8) as usize];
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
pub(super) fn is_phase_slice_uniform(p_slice: &[PhaseQ7], num_heads: usize) -> bool {
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

#[inline(always)]
pub(super) fn apply_phase_steering_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[PhaseQ7],
    head_dim: usize,
    num_heads: usize,
) {
    apply_phase_steering_slice_core(q_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_slice_core(k_slice, p_slice, head_dim, num_heads);
}

pub(crate) fn rotate_phase_clamp_upper_slice(buf: &mut [f32], sin_t: f32, cos_t: f32) {
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
                avx2_rotate_so2_block32(ptr, v_cos, v_sin);
                return;
            }
            if len == 64 {
                avx2_rotate_so2_block32(ptr, v_cos, v_sin);
                avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
                return;
            }
        }
    }
    rotate_so2_slice_unrolled16(buf, sin_t, cos_t);
}

#[inline]
pub(crate) fn rotate_phase_clamp_upper_heads_uniform_sincos(
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
        rotate_phase_clamp_upper_slice(
            &mut t_slice[base + upper_off..base + head_dim],
            sin_t,
            cos_t,
        );
        h += 1;
    }
}

/// Rotates strictly the upper low-frequency quarter `rot_dim = ((head_dim >> 2) & !1).max(2)`
/// (`[base + head_dim - rot_dim .. base + head_dim]`) of each head in `t_slice` with `PhaseClamp` limiting,
/// hoisting `rot_dim == 16 / 32 / 64` specialization outside the head loop and coalescing consecutive
/// uniform 8-head SWAR cells.
#[inline(never)]
pub(super) fn apply_phase_clamp_single_slice_core(
    t_slice: &mut [f32],
    p_slice: &[PhaseQ7],
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
) {
    let rot_dim = phase_clamp_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    let (is_default, eff_max, q7_to_norm_scale) = resolve_phase_clamp_norm_params(max_rad);

    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF
        } else {
            Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF
        };
        if num_heads >= 8 && is_phase_slice_uniform(p_slice, num_heads) {
            let b0 = p_slice[0] as u8;
            if b0 != 0 {
                let (sin_t, cos_t) = lut[b0 as usize];
                rotate_phase_clamp_upper_heads_uniform_sincos(
                    t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
                );
            }
            return;
        }
        apply_phase_clamp_default_nonuniform_slice_core(
            t_slice, p_slice, head_dim, num_heads, rot_dim, upper_off, lut,
        );
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
                let sc = branchless_sincos_q7_phase_clamp_preinv(eff_q7, eff_max, q7_to_norm_scale);
                cached_q7 = q7_i32;
                cached_sincos = sc;
                sc
            };
            rotate_phase_clamp_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
        }
    }
}

/// Rotates strictly the upper low-frequency quarter (`rot_pairs = phase_clamp_upper_rot_dim(rotary_dim) / 2`)
/// of a HalfSplit (NeoX) tensor `t_slice` across `num_heads` heads of stride `head_dim`, leaving
/// non-RoPE dimensions `[rotary_dim .. head_dim]` untouched.
#[inline(never)]
pub(super) fn apply_phase_clamp_half_split_slice_core(
    t_slice: &mut [f32],
    p_slice: &[PhaseQ7],
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
    max_rad: f32,
) {
    let half_rot = rotary_dim >> 1;
    let rot_pairs = (phase_clamp_upper_rot_dim(rotary_dim) >> 1).min(half_rot);
    let start = half_rot - rot_pairs;
    let total_len = num_heads * head_dim;
    let t_slice = &mut t_slice[..total_len];
    let p_slice = &p_slice[..num_heads];
    let (is_default, eff_max, q7_to_norm_scale) = resolve_phase_clamp_norm_params(max_rad);

    let mut cached_q7: i32 = i32::MIN;
    let mut cached_sincos = (0.0_f32, 1.0_f32);
    for (head_buf, &raw_q7) in t_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
        let q7_i32 = i32::from(raw_q7);
        let (sin_t, cos_t) = if q7_i32 == cached_q7 {
            cached_sincos
        } else {
            let sc = if is_default {
                Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF[raw_q7 as u8 as usize]
            } else {
                branchless_sincos_q7_phase_clamp_preinv(raw_q7, eff_max, q7_to_norm_scale)
            };
            cached_q7 = q7_i32;
            cached_sincos = sc;
            sc
        };
        if sin_t == 0.0 && cos_t == 1.0 {
            continue;
        }
        let (h1, h2) = head_buf[..rotary_dim].split_at_mut(half_rot);
        let x1 = &mut h1[start..half_rot];
        let x2 = &mut h2[start..half_rot];
        for (a, b) in x1.iter_mut().zip(x2.iter_mut()) {
            let va = *a;
            let vb = *b;
            *a = va * cos_t - vb * sin_t;
            *b = va * sin_t + vb * cos_t;
        }
    }
}

/// Computes per-head Q7 Procrustes phase deltas directly on HalfSplit `(x[i], x[i + rotary_dim/2])`
/// slices of stride `head_dim` without intermediate layout conversion.
#[inline(never)]
pub(super) fn procrustes_q7_half_split_slice(
    src_slice: &[f32],
    tgt_slice: &[f32],
    head_dim: usize,
    rotary_dim: usize,
    num_heads: usize,
    phases_out: &mut [PhaseQ7],
) {
    let limit_heads = num_heads.min(MAX_ATTENTION_HEADS).min(phases_out.len());
    let half_rot = rotary_dim >> 1;
    let mut h_base = 0usize;
    while h_base < limit_heads {
        let cell_heads = (limit_heads - h_base).min(8);
        let mut sin_arr = [0.0_f32; 8];
        let mut cos_arr = [0.0_f32; 8];
        let mut lane = 0usize;
        while lane < cell_heads {
            let base = (h_base + lane) * head_dim;
            let s1 = &src_slice[base..base + half_rot];
            let s2 = &src_slice[base + half_rot..base + rotary_dim];
            let t1 = &tgt_slice[base..base + half_rot];
            let t2 = &tgt_slice[base + half_rot..base + rotary_dim];
            let mut sum_cos = 0.0_f32;
            let mut sum_sin = 0.0_f32;
            for i in 0..half_rot {
                sum_cos += s1[i] * t1[i] + s2[i] * t2[i];
                sum_sin += s1[i] * t2[i] - s2[i] * t1[i];
            }
            cos_arr[lane] = sum_cos;
            sin_arr[lane] = sum_sin;
            lane += 1;
        }
        let word = atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr);
        let mut w = word;
        let mut out_lane = 0usize;
        while out_lane < cell_heads {
            phases_out[h_base + out_lane] = (w & 0xFF) as PhaseQ7;
            w >>= 8;
            out_lane += 1;
        }
        h_base += cell_heads;
    }
}

mod swar_dispatch;
pub(super) use swar_dispatch::{
    apply_phase_clamp_default_nonuniform_slice_core, apply_phase_clamp_pair_slice_core,
};
pub(crate) use swar_dispatch::{
    steer_from_swar_words_core_safe, steer_single_slice_from_eff_words_core,
};
