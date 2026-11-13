// name: src/core/src/traits/rope/kernels.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core SWAR/AVX2 RoPE rotation, RIPA, SLERP, Procrustes, and Bayesian attention helpers.

use super::*;

/// Computes the RIPA / PSPM upper low-frequency rotation dimension
/// `rot_dim = ((head_dim / 4) & !1).max(2)` (even integer in `2..=head_dim`).
#[inline(always)]
#[must_use]
pub const fn ripa_upper_rot_dim(head_dim: usize) -> usize {
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
///   or persona override, or when Toroidal $k$-Means concentration is ultra-sharp ($\kappa_c \ge 6.0$),
///   preserving high- and mid-frequency syntactic/lexical RoPE channels.
/// - **Baseline RIPA Aperture ($\frac{1}{4}d_{\mathrm{head}}$)**: Standard upper-quarter rotation
///   (`ripa_upper_rot_dim(head_dim)`).
/// - **Expanded Persona / High-Quorum Aperture ($\frac{3}{8}d_{\mathrm{head}}$)**: Selected when
///   Subjective Persona grounding (`CAUSAL_AXIS_SUBJECTIVE_PERSONA`, bit `2`) resonates with high
///   Micro-Swarm quorum (`quorum_mask.count_ones() >= 6`) and moderate concentration, allowing
///   persona voice and cadence to steer mid-frequency syntactic phrasing.
/// - **Wide Repulsion / Bifurcation Escape Aperture ($\frac{1}{2}d_{\mathrm{head}}$)**: Selected
///   when Contradiction Guard repulsion (`CAUSAL_AXIS_CONTRADICTION_GUARD`, bit `3`) or an EDBC
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
pub(crate) const fn ptr_has_action_mode(tagged_ptr: gcso_tagged_ptr_t, mode_mask: u8) -> bool {
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
pub(crate) const fn eval_ripa_sincos_with_params(
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

/// Rotates `t_slice` across `num_heads` heads without branching on individual phase magnitudes,
/// coalescing uniform phase profiles (`is_phase_slice_uniform`) into a single-broadcast AVX2
/// streaming pass when all active heads share the same Q0.7 angle.
#[inline(never)]
pub(super) fn apply_phase_steering_slice_core(
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
pub(super) fn is_phase_slice_uniform(p_slice: &[gcso_q7_t], num_heads: usize) -> bool {
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
pub(super) fn unpack_sincos_heads_nonuniform_64(
    p_slice: &[gcso_q7_t],
    num_heads: usize,
    sincos_out: &mut [(f32, f32); MAX_ATTENTION_HEADS],
) {
    let lut = Q7_SINCOS_LUT_256_REF;
    let mut h = 0usize;
    while h + 8 <= num_heads {
        let mut w = load_q7_x8_slice(p_slice, h);
        let b0 = (w & 0xFF) as u8;
        if w == w.rotate_right(8) {
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
pub(super) fn phase_rows_equal_swar(row_a: &[gcso_q7_t], row_b: &[gcso_q7_t], num_heads: usize) -> bool {
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
pub(super) fn is_batch_phase_row_invariant(
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
pub(super) fn prefetch_next_token_pages(next_tok_ptr: *const f32, token_elems: usize) {
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
pub(super) fn apply_phase_steering_multi_token_batch_core(
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
pub(super) fn apply_phase_steering_fused_multi_token_batch_core(
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
pub(super) fn apply_phase_steering_fused_safe_multi_token_batch_core(
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
#[inline]
pub(super) fn apply_phase_steering_slice_with_sincos(
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
            // Unroll 1 head (`4x avx2_rotate_so2_block32`) per iteration to keep the loop body
            // compact for instruction-cache / decoded-uop-cache residency.
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
            // Single flat loop over `num_heads * 2` 128-float blocks (`blk >> 1` phase index),
            // keeping the inner loop body compact for instruction-cache / decoded-uop-cache residency.
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
#[inline]
pub(super) fn apply_phase_steering_pair_slice_with_sincos(
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
            apply_phase_steering_slice_with_sincos(
                q_slice,
                sincos_heads,
                false,
                head_dim,
                num_heads,
            );
            apply_phase_steering_slice_with_sincos(
                k_slice,
                sincos_heads,
                false,
                head_dim,
                num_heads,
            );
            return;
        }
        if head_dim == 32 {
            // SAFETY: `q_slice` and `k_slice` have length `num_heads * 32` and AVX2 is enabled.
            unsafe {
                let mut h = 0usize;
                while h + 2 <= num_heads {
                    let (sin0, cos0) = *sincos_heads.get_unchecked(h);
                    avx2_rotate_so2_pair_block32(
                        qp,
                        kp,
                        _mm256_set1_ps(cos0),
                        _mm256_set1_ps(sin0),
                    );
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

#[inline(always)]
pub(super) fn apply_phase_steering_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    apply_phase_steering_slice_core(q_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_slice_core(k_slice, p_slice, head_dim, num_heads);
}

#[inline(always)]
pub(super) fn apply_phase_steering_triple_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    v_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
) {
    apply_phase_steering_slice_core(q_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_slice_core(k_slice, p_slice, head_dim, num_heads);
    apply_phase_steering_slice_core(v_slice, p_slice, head_dim, num_heads);
}

pub(crate) fn rotate_ripa_upper_slice(buf: &mut [f32], sin_t: f32, cos_t: f32) {
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
pub(crate) fn rotate_ripa_upper_heads_uniform_sincos(
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
pub(super) fn unpack_ripa_sincos_heads_64(
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
#[inline]
pub(super) fn apply_ripa_single_slice_with_sincos(
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
        rotate_ripa_upper_heads_uniform_sincos(
            t_slice, 0, num_heads, head_dim, rot_dim, sin_t, cos_t,
        );
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
pub(super) fn apply_ripa_single_slice_core(
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
        apply_ripa_default_nonuniform_slice_core(
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
pub(super) unsafe fn apply_ripa_default_nonuniform_avx2_ptr(
    ptr: *mut f32,
    p_ptr: *const u8,
    head_dim: usize,
    num_heads_and_geo_and_rot: usize,
) {
    use core::arch::x86_64::{
        _mm256_addsub_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_permute_ps, _mm256_set1_ps,
        _mm256_storeu_ps,
    };
    let fold_geodesic = (num_heads_and_geo_and_rot >> 31) != 0;
    let num_heads = num_heads_and_geo_and_rot & 0xFFFF;
    let packed_rot = (num_heads_and_geo_and_rot >> 16) & 0x7FFF;
    let rot_dim = if packed_rot != 0 {
        packed_rot
    } else {
        ripa_upper_rot_dim(head_dim)
    };
    let upper_off = head_dim - rot_dim;
    let lut = if fold_geodesic {
        Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
    } else {
        Q7_DEFAULT_RIPA_SINCOS_REF
    };
    // SAFETY: Caller guarantees `ptr` is valid for `num_heads * head_dim` floats and AVX2 is enabled.
    unsafe {
        if rot_dim == 32 {
            let mut hp = ptr.add(upper_off);
            let step2 = head_dim * 2;
            let step3 = head_dim * 3;
            let step4 = head_dim * 4;
            let mut h = 0usize;
            let aligned8_end = num_heads & !7usize;
            while h + 4 <= num_heads {
                let wb32 = if h < aligned8_end {
                    let w64 = p_ptr.add(h & !7usize).cast::<u64>().read_unaligned();
                    (w64 >> ((h & 4) << 3)) as u32
                } else {
                    p_ptr.add(h).cast::<u32>().read_unaligned()
                };
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
                let wb16 =
                    (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 6) << 3))
                        as u32;
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
                let (s1, c1) = lut[((wb16 >> 8) as u8) as usize];
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
                let b_h =
                    (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 7) << 3)) as u8;
                let (sin_t, cos_t) = lut[b_h as usize];
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
                let wb16 =
                    (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 6) << 3))
                        as u32;
                let (s0, c0) = lut[(wb16 as u8) as usize];
                let vc0 = _mm256_set1_ps(c0);
                let vs0 = _mm256_set1_ps(s0);
                avx2_rotate_so2_block32(hp, vc0, vs0);
                avx2_rotate_so2_block32(hp.add(32), vc0, vs0);
                let hp1 = hp.add(head_dim);
                let (s1, c1) = lut[((wb16 >> 8) as u8) as usize];
                let vc1 = _mm256_set1_ps(c1);
                let vs1 = _mm256_set1_ps(s1);
                avx2_rotate_so2_block32(hp1, vc1, vs1);
                avx2_rotate_so2_block32(hp1.add(32), vc1, vs1);
                hp = hp.add(step2);
                h += 2;
            }
            while h < num_heads {
                let b_h =
                    (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 7) << 3)) as u8;
                let (sin_t, cos_t) = lut[b_h as usize];
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
pub(super) fn apply_ripa_default_nonuniform_slice_core(
    t_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    lut: &[(f32, f32); 256],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if num_heads <= 0xFFFF && rot_dim <= 0x7FFF {
        let _ = upper_off;
        let is_geo = core::ptr::eq(lut, Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF);
        let packed = num_heads | (rot_dim << 16) | ((is_geo as usize) << 31);
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
    let _ = (num_heads, rot_dim);
    for (head_buf, &raw_q7) in t_slice.chunks_exact_mut(head_dim).zip(p_slice.iter()) {
        let (sin_t, cos_t) = lut[(raw_q7 as u8) as usize];
        rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
    }
}

#[inline(always)]
pub(super) fn apply_ripa_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[gcso_q7_t],
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
) {
    apply_ripa_single_slice_core(q_slice, p_slice, head_dim, num_heads, fold_geodesic, max_rad);
    apply_ripa_single_slice_core(k_slice, p_slice, head_dim, num_heads, fold_geodesic, max_rad);
}

#[inline]
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
    if cell_heads == 0 || raw_word == 0 {
        return;
    }
    let base = head_start * head_dim;
    let span = cell_heads * head_dim;
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_RIPA_SINCOS_REF
        } else {
            Q7_DEFAULT_RIPA_SINCOS_REF
        };
        let b0 = (raw_word & 0xFF) as u8;
        if raw_word == raw_word.rotate_right(8) {
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
        let mut w = raw_word;
        for head_buf in t_slice[base..base + span].chunks_exact_mut(head_dim) {
            let b = (w & 0xFF) as u8;
            w >>= 8;
            if b != 0 {
                let (sin_t, cos_t) = lut[b as usize];
                rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
            }
        }
    } else {
        let mut w = if fold_geodesic {
            swar_q7_fold_s1_geodesic_x8(raw_word)
        } else {
            raw_word
        };
        let b0 = (w & 0xFF) as u8;
        if w == w.rotate_right(8) {
            if b0 != 0 {
                let (sin_t, cos_t) = eval_ripa_sincos_with_params(
                    b0 as gcso_q7_t,
                    false,
                    is_default,
                    eff_max,
                    q7_to_norm_scale,
                );
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
        let mut cached_q7 = i32::MIN;
        let mut cached_sincos = (0.0_f32, 1.0_f32);
        for head_buf in t_slice[base..base + span].chunks_exact_mut(head_dim) {
            let q7_val = (w & 0xFF) as gcso_q7_t;
            w >>= 8;
            if q7_val != 0 {
                let q7_i32 = i32::from(q7_val);
                let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                    cached_sincos
                } else {
                    let sc = eval_ripa_sincos_with_params(
                        q7_val,
                        false,
                        is_default,
                        eff_max,
                        q7_to_norm_scale,
                    );
                    cached_q7 = q7_i32;
                    cached_sincos = sc;
                    sc
                };
                rotate_ripa_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
            }
        }
    }
}

pub(crate) fn steer_single_slice_from_eff_words_core(
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
            if w0 == w0.rotate_right(8) {
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
            if num_heads <= 0xFFFF && rot_dim <= 0x7FFF {
                let packed = num_heads | (rot_dim << 16) | ((fold_geodesic as usize) << 31);
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
            if w == w.rotate_right(8) {
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
            if w0 == w0.rotate_right(8) {
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
            if raw_w == raw_w.rotate_right(8) {
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
pub(super) fn steer_pair_slice_from_eff_words_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    eff_words: &[u64; NUM_SWARM_CELLS],
    run_mask: u8,
    cell_limit_mask: u8,
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    fold_geodesic: bool,
    is_default: bool,
    eff_max: f32,
    q7_to_norm_scale: f32,
) {
    steer_single_slice_from_eff_words_core(
        q_slice,
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
    steer_single_slice_from_eff_words_core(
        k_slice,
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
pub(super) fn eval_multi_head_phase_coherence_unrolled4(q_p: &[gcso_q7_t], m_p: &[gcso_q7_t]) -> f32 {
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
        } else if dw == dw.rotate_right(8) {
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

