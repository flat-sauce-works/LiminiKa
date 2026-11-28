// name: src/core/src/foundation/rope/kernels/swar_dispatch.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! SWAR word-packed and AVX2 PhaseClamp steering dispatch kernels.

use super::*;

/// Rotates strictly the upper low-frequency quarter `rot_dim` of a single tensor `t_slice` across
/// `num_heads` heads using a preselected default `PhaseClamp` LUT (`Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF` or
/// `Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF`) when `p_slice` is already known to be non-uniform.
/// Kept `#[inline(never)]` so single-tensor and fused-QK `head_dim == 128` callers share the exact
/// same 512B Decoded ICache (DSB) loop without `for base_ptr in [q, k]` 2x loop-unroll code bloat.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(never)]
pub(in crate::foundation::rope) unsafe fn apply_phase_clamp_default_nonuniform_avx2_ptr(
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
        phase_clamp_upper_rot_dim(head_dim)
    };
    let upper_off = head_dim - rot_dim;
    let lut = if fold_geodesic {
        Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF
    } else {
        Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF
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
                let wb16 = (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 6) << 3))
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
                let wb16 = (p_ptr.add(h & !7usize).cast::<u64>().read_unaligned() >> ((h & 6) << 3))
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
pub(in crate::foundation::rope) fn apply_phase_clamp_default_nonuniform_slice_core(
    t_slice: &mut [f32],
    p_slice: &[PhaseQ7],
    head_dim: usize,
    num_heads: usize,
    rot_dim: usize,
    upper_off: usize,
    lut: &[(f32, f32); 256],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if num_heads <= 0xFFFF && rot_dim <= 0x7FFF {
        let _ = upper_off;
        let is_geo = core::ptr::eq(lut, Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF);
        let packed = num_heads | (rot_dim << 16) | ((is_geo as usize) << 31);
        // SAFETY: Caller has validated `t_slice` and `p_slice` lengths.
        unsafe {
            apply_phase_clamp_default_nonuniform_avx2_ptr(
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
        rotate_phase_clamp_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
    }
}

#[inline(always)]
pub(in crate::foundation::rope) fn apply_phase_clamp_pair_slice_core(
    q_slice: &mut [f32],
    k_slice: &mut [f32],
    p_slice: &[PhaseQ7],
    head_dim: usize,
    num_heads: usize,
    fold_geodesic: bool,
    max_rad: f32,
) {
    apply_phase_clamp_single_slice_core(
        q_slice,
        p_slice,
        head_dim,
        num_heads,
        fold_geodesic,
        max_rad,
    );
    apply_phase_clamp_single_slice_core(
        k_slice,
        p_slice,
        head_dim,
        num_heads,
        fold_geodesic,
        max_rad,
    );
}

#[inline]
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(crate) fn steer_phase_clamp_single_cell_from_word_core(
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
            Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF
        } else {
            Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF
        };
        let b0 = (raw_word & 0xFF) as u8;
        if raw_word == raw_word.rotate_right(8) {
            if b0 != 0 {
                let (sin_t, cos_t) = lut[b0 as usize];
                rotate_phase_clamp_upper_heads_uniform_sincos(
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
                rotate_phase_clamp_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
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
                let (sin_t, cos_t) = eval_phase_clamp_sincos_with_params(
                    b0 as PhaseQ7,
                    false,
                    is_default,
                    eff_max,
                    q7_to_norm_scale,
                );
                rotate_phase_clamp_upper_heads_uniform_sincos(
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
            let q7_val = (w & 0xFF) as PhaseQ7;
            w >>= 8;
            if q7_val != 0 {
                let q7_i32 = i32::from(q7_val);
                let (sin_t, cos_t) = if q7_i32 == cached_q7 {
                    cached_sincos
                } else {
                    let sc = eval_phase_clamp_sincos_with_params(
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
                rotate_phase_clamp_upper_slice(&mut head_buf[upper_off..], sin_t, cos_t);
            }
        }
    }
}

// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(crate) fn steer_single_slice_from_eff_words_core(
    t_slice: &mut [f32],
    eff_words: &[u64; NUM_SWARM_CELLS],
    mut run_mask: u8,
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
    let active_cells = num_heads.div_ceil(HEADS_PER_CELL).min(NUM_SWARM_CELLS);
    let max_cell_mask = ((1u16 << active_cells) - 1) as u8;
    let cell_limit_mask = cell_limit_mask & max_cell_mask;
    run_mask &= cell_limit_mask;
    if run_mask == 0 || num_heads == 0 {
        return;
    }
    if is_default {
        let lut = if fold_geodesic {
            Q7_GEODESIC_DEFAULT_PHASE_CLAMP_SINCOS_REF
        } else {
            Q7_DEFAULT_PHASE_CLAMP_SINCOS_REF
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
                    if run_mask == cell_limit_mask {
                        rotate_phase_clamp_upper_heads_uniform_sincos(
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
                        rotate_phase_clamp_upper_heads_uniform_sincos(
                            t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                        );
                    }
                }
                return;
            }
        }
        if run_mask == cell_limit_mask {
            #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
            if num_heads <= 0xFFFF && rot_dim <= 0x7FFF {
                let packed = num_heads | (rot_dim << 16) | ((fold_geodesic as usize) << 31);
                // SAFETY: `t_slice` has length `num_heads * head_dim`, `eff_words` has `8 * 8 = 64 >= num_heads`
                // little-endian Q7 bytes, and AVX2 is enabled. Delegating to the shared frameless leaf
                // `apply_phase_clamp_default_nonuniform_avx2_ptr` keeps all swarm/SLERP/unified-manifold callers
                // 100% GPR-only (`0` Windows x64 `xmm6..xmm15` stack saves) and DSB-resident.
                unsafe {
                    apply_phase_clamp_default_nonuniform_avx2_ptr(
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
                    rotate_phase_clamp_upper_heads_uniform_sincos(
                        t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                continue;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            steer_phase_clamp_single_cell_from_word_core(
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
        if run_mask == cell_limit_mask && is_word_array_uniform_64(eff_words) {
            let w0 = eff_words[0];
            let b0 = (w0 & 0xFF) as u8;
            if w0 == w0.rotate_right(8) {
                if b0 != 0 {
                    let eff_q7 = if fold_geodesic {
                        branchless_q7_fold_s1_geodesic(b0 as PhaseQ7)
                    } else {
                        b0 as PhaseQ7
                    };
                    let (sin_t, cos_t) =
                        branchless_sincos_q7_phase_clamp_preinv(eff_q7, eff_max, q7_to_norm_scale);
                    rotate_phase_clamp_upper_heads_uniform_sincos(
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
                        branchless_q7_fold_s1_geodesic(b0 as PhaseQ7)
                    } else {
                        b0 as PhaseQ7
                    };
                    let (sin_t, cos_t) =
                        branchless_sincos_q7_phase_clamp_preinv(eff_q7, eff_max, q7_to_norm_scale);
                    rotate_phase_clamp_upper_heads_uniform_sincos(
                        t_slice, head_start, head_end, head_dim, rot_dim, sin_t, cos_t,
                    );
                }
                continue;
            }
            let cell_heads = (num_heads - head_start).min(HEADS_PER_CELL);
            steer_phase_clamp_single_cell_from_word_core(
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
// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
pub(in crate::foundation::rope) fn steer_pair_slice_from_eff_words_core(
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

// Hot-path register-passed SIMD/SWAR kernel leaf kept flat to avoid stack-frame spills.
#[allow(clippy::too_many_arguments)]
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

    let rot_dim = phase_clamp_upper_rot_dim(head_dim);
    let upper_off = head_dim - rot_dim;
    let full_consensus = (eff_quorum == cell_limit_mask) | (clean_damping_u7 >= 128);
    let damping_s = u64::from(clean_damping_u7);
    let damping_bias = (128u64.wrapping_sub(damping_s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
    let (is_default, eff_max, q7_to_norm_scale) = resolve_phase_clamp_norm_params(max_rad);

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
