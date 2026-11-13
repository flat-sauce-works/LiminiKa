// name: src/core/src/traits/rope/subhead_routing.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parallel Sub-Head Phase Modulation (PspmRouter) trait and dispatch helpers.

use super::*;

// ===================================================================
// 4. Parallel Sub-Head Phase Modulation (`PspmRouter`)
// ===================================================================

#[inline(always)]
#[must_use]
pub(super) const fn sanitize_pspm_gain_to_u7(gain: f32) -> u8 {
    let clean_gain = branchless_select_f32(branchless_is_finite_f32(gain), gain, 1.0);
    (branchless_clamp_f32(clean_gain, 0.0, 2.0) * 128.0 + 0.5) as u8
}

#[inline(always)]
#[must_use]
pub(super) const fn compute_pspm_eff_q7(
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
pub(super) const fn compute_pspm_eff_word_x8(
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
    } else if gain_u7 < 128 {
        swar_q7_scale_le128_x8(base_w, gain_u7 as u64)
    } else {
        swar_q7_scale_u7_x8(base_w, gain_u7)
    };
    if pulse_q7 == 0 {
        return if fold_geodesic & (gain_u7 > 128) {
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
pub(super) fn dispatch_subhead_profiles_core(
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
    if head_dim == 0 || (head_dim & 1) != 0 || max_rad <= 0.0 || !branchless_is_finite_f32(max_rad)
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
#[inline]
#[must_use]
pub(super) const fn build_pspm_eff_words_64(
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
                let ow =
                    compute_pspm_eff_word_x8(iw, explore_gain_u7, explore_pulse_q7, fold_geodesic);
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
#[inline]
#[must_use]
pub(super) fn build_pspm_eff_words_from_profiles_64(
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
                let ow =
                    compute_pspm_eff_word_x8(iw, explore_gain_u7, explore_pulse_q7, fold_geodesic);
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
pub(super) const fn byte_uniform_word_u8(w: &[u64; NUM_SWARM_CELLS]) -> Option<u8> {
    let w0 = w[0];
    let b = (w0 & 0xFF) as u8;
    if w0 == w0.rotate_right(8) && is_word_array_uniform_64(w) {
        Some(b)
    } else {
        None
    }
}

/// Evaluates the fused $SO(2)$ group-closure `(sin, cos)` for a single `(b_u8, s_u8)` Q7 pair,
/// fast-pathing default RIPA clamps (`0.24 rad`) via `.rodata` LUTs.
#[inline(always)]
#[must_use]
pub(super) fn eval_fused_ripa_pspm_pair_u8(
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
#[inline]
pub(super) fn rotate_fused_ripa_and_pspm_uniform_subgroups(
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
    let eff_f = if fh > 0 {
        f_u8
    } else if le > 0 {
        l_u8
    } else {
        e_u8
    };
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
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, 0, limit, head_dim, rot_dim, sin_c, cos_c,
            );
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
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, 0, le, head_dim, rot_dim, sin_fl, cos_fl,
            );
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
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, le, limit, head_dim, rot_dim, sin_e, cos_e,
            );
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
            rotate_ripa_upper_heads_uniform_sincos(
                t_slice, fh, limit, head_dim, rot_dim, sin_le, cos_le,
            );
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

#[inline(always)]
pub(super) fn rotate_ripa_lut_words_slice_nonuniform(
    t_slice: &mut [f32],
    words: &[u64; NUM_SWARM_CELLS],
    limit: usize,
    head_dim: usize,
    rot_dim: usize,
    lut: &[(f32, f32); 256],
) {
    let upper_off = head_dim - rot_dim;
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    if rot_dim == 32 && t_slice.len() >= limit.saturating_mul(head_dim) {
        // SAFETY: `t_slice.len() >= limit * head_dim`, `rot_dim == 32 <= head_dim`, and `avx2` is enabled.
        unsafe {
            use core::arch::x86_64::_mm256_set1_ps;
            let mut hp = t_slice.as_mut_ptr().add(upper_off);
            let step2 = head_dim * 2;
            let step3 = head_dim * 3;
            let step4 = head_dim * 4;
            let mut h = 0usize;
            while h + 4 <= limit {
                let wb32 = (words[h >> 3] >> ((h & 4) << 3)) as u32;
                if wb32 != 0 {
                    let b0 = (wb32 & 0xFF) as usize;
                    if b0 != 0 {
                        let (s0, c0) = lut[b0];
                        avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                    }
                    let b1 = ((wb32 >> 8) & 0xFF) as usize;
                    if b1 != 0 {
                        let (s1, c1) = lut[b1];
                        avx2_rotate_so2_block32(
                            hp.add(head_dim),
                            _mm256_set1_ps(c1),
                            _mm256_set1_ps(s1),
                        );
                    }
                    let b2 = ((wb32 >> 16) & 0xFF) as usize;
                    if b2 != 0 {
                        let (s2, c2) = lut[b2];
                        avx2_rotate_so2_block32(
                            hp.add(step2),
                            _mm256_set1_ps(c2),
                            _mm256_set1_ps(s2),
                        );
                    }
                    let b3 = (wb32 >> 24) as usize;
                    if b3 != 0 {
                        let (s3, c3) = lut[b3];
                        avx2_rotate_so2_block32(
                            hp.add(step3),
                            _mm256_set1_ps(c3),
                            _mm256_set1_ps(s3),
                        );
                    }
                }
                hp = hp.add(step4);
                h += 4;
            }
            while h < limit {
                let b = ((words[h >> 3] >> ((h & 7) << 3)) & 0xFF) as usize;
                if b != 0 {
                    let (sin_t, cos_t) = lut[b];
                    avx2_rotate_so2_block32(hp, _mm256_set1_ps(cos_t), _mm256_set1_ps(sin_t));
                }
                hp = hp.add(head_dim);
                h += 1;
            }
        }
        return;
    }
    let mut h = 0usize;
    while h < limit {
        let b = ((words[h >> 3] >> ((h & 7) << 3)) & 0xFF) as u8;
        if b != 0 {
            let (sin_t, cos_t) = lut[b as usize];
            let base = h * head_dim + upper_off;
            rotate_ripa_upper_slice(&mut t_slice[base..base + rot_dim], sin_t, cos_t);
        }
        h += 1;
    }
}

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

#[inline]
pub(super) fn rotate_fused_ripa_and_pspm_eff_words_core(
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
        let self_lut = Q7_FUSED_SELF_DEFAULT_RIPA_SINCOS_REF;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if rot_dim == 32 && t_slice.len() >= limit.saturating_mul(head_dim) {
            // SAFETY: `t_slice.len() >= limit * head_dim`, `rot_dim == 32 <= head_dim`, and `avx2` is enabled.
            unsafe {
                use core::arch::x86_64::_mm256_set1_ps;
                let mut hp = t_slice.as_mut_ptr().add(upper_off);
                let step2 = head_dim * 2;
                let step3 = head_dim * 3;
                let step4 = head_dim * 4;
                let mut h = 0usize;
                while h + 4 <= limit {
                    let cell = h >> 3;
                    let half_shift = (h & 4) << 3;
                    let wb32 = (base_ripa_words[cell] >> half_shift) as u32;
                    let ws32 = (pspm_eff_words[cell] >> half_shift) as u32;
                    if (wb32 | ws32) != 0 {
                        if wb32 == ws32 {
                            let b0 = (wb32 & 0xFF) as usize;
                            if b0 != 0 {
                                let (s0, c0) = self_lut[b0];
                                avx2_rotate_so2_block32(hp, _mm256_set1_ps(c0), _mm256_set1_ps(s0));
                            }
                            let b1 = ((wb32 >> 8) & 0xFF) as usize;
                            if b1 != 0 {
                                let (s1, c1) = self_lut[b1];
                                avx2_rotate_so2_block32(
                                    hp.add(head_dim),
                                    _mm256_set1_ps(c1),
                                    _mm256_set1_ps(s1),
                                );
                            }
                            let b2 = ((wb32 >> 16) & 0xFF) as usize;
                            if b2 != 0 {
                                let (s2, c2) = self_lut[b2];
                                avx2_rotate_so2_block32(
                                    hp.add(step2),
                                    _mm256_set1_ps(c2),
                                    _mm256_set1_ps(s2),
                                );
                            }
                            let b3 = (wb32 >> 24) as usize;
                            if b3 != 0 {
                                let (s3, c3) = self_lut[b3];
                                avx2_rotate_so2_block32(
                                    hp.add(step3),
                                    _mm256_set1_ps(c3),
                                    _mm256_set1_ps(s3),
                                );
                            }
                        } else {
                            let mut lane = 0usize;
                            while lane < 4 {
                                let b_u8 = ((wb32 >> (lane * 8)) & 0xFF) as usize;
                                let s_u8 = ((ws32 >> (lane * 8)) & 0xFF) as usize;
                                if (b_u8 | s_u8) != 0 {
                                    let (sin_comb, cos_comb) = if b_u8 == s_u8 {
                                        self_lut[b_u8]
                                    } else {
                                        let (sb, cb) = lut[b_u8];
                                        let (sp, cp) = lut[s_u8];
                                        (sb * cp + cb * sp, cb * cp - sb * sp)
                                    };
                                    avx2_rotate_so2_block32(
                                        hp.add(lane * head_dim),
                                        _mm256_set1_ps(cos_comb),
                                        _mm256_set1_ps(sin_comb),
                                    );
                                }
                                lane += 1;
                            }
                        }
                    }
                    hp = hp.add(step4);
                    h += 4;
                }
                while h < limit {
                    let cell = h >> 3;
                    let shift = (h & 7) << 3;
                    let b_u8 = ((base_ripa_words[cell] >> shift) & 0xFF) as usize;
                    let s_u8 = ((pspm_eff_words[cell] >> shift) & 0xFF) as usize;
                    if (b_u8 | s_u8) != 0 {
                        let (sin_comb, cos_comb) = if b_u8 == s_u8 {
                            self_lut[b_u8]
                        } else {
                            let (sb, cb) = lut[b_u8];
                            let (sp, cp) = lut[s_u8];
                            (sb * cp + cb * sp, cb * cp - sb * sp)
                        };
                        avx2_rotate_so2_block32(
                            hp,
                            _mm256_set1_ps(cos_comb),
                            _mm256_set1_ps(sin_comb),
                        );
                    }
                    hp = hp.add(head_dim);
                    h += 1;
                }
            }
            return;
        }
        let mut h = 0usize;
        while h < limit {
            let cell = h >> 3;
            let shift = (h & 7) << 3;
            let b_u8 = ((base_ripa_words[cell] >> shift) & 0xFF) as u8;
            let s_u8 = ((pspm_eff_words[cell] >> shift) & 0xFF) as u8;
            if (b_u8 | s_u8) != 0 {
                let (sin_comb, cos_comb) = if b_u8 == s_u8 {
                    self_lut[b_u8 as usize]
                } else {
                    let (sb, cb) = lut[b_u8 as usize];
                    let (sp, cp) = lut[s_u8 as usize];
                    (sb * cp + cb * sp, cb * cp - sb * sp)
                };
                let base = h * head_dim + upper_off;
                rotate_ripa_upper_slice(&mut t_slice[base..base + rot_dim], sin_comb, cos_comb);
            }
            h += 1;
        }
        return;
    }

    let mut cached_pair = u16::MAX;
    let mut cached_sc = (0.0_f32, 1.0_f32);
    let mut h = 0usize;
    while h < limit {
        let cell = h >> 3;
        let shift = (h & 7) << 3;
        let b_u8 = ((base_ripa_words[cell] >> shift) & 0xFF) as u8;
        let s_u8 = ((pspm_eff_words[cell] >> shift) & 0xFF) as u8;
        if (b_u8 | s_u8) != 0 {
            let pair_key = (u16::from(b_u8) << 8) | u16::from(s_u8);
            let (sin_comb, cos_comb) = if pair_key == cached_pair {
                cached_sc
            } else {
                let sc = eval_fused_ripa_pspm_pair_u8(
                    b_u8,
                    s_u8,
                    base_is_def,
                    base_eff_max,
                    base_scale,
                    pspm_is_def,
                    pspm_eff_max,
                    pspm_scale,
                );
                cached_pair = pair_key;
                cached_sc = sc;
                sc
            };
            let base = h * head_dim + upper_off;
            rotate_ripa_upper_slice(&mut t_slice[base..base + rot_dim], sin_comb, cos_comb);
        }
        h += 1;
    }
}

#[inline]
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

    let fact_is_self = (fact_gain_u7 == 128)
        && (core::ptr::eq(fact_words, base_ripa_words) || fact_words == base_ripa_words);
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
            Some(0u8)
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
            let f_eff_u8 =
                compute_pspm_eff_q7(fb as gcso_q7_t, fact_gain_u7, 0, fold_geodesic) as u8;
            let l_eff_u8 =
                compute_pspm_eff_q7(lb as gcso_q7_t, logic_gain_u7, 0, fold_geodesic) as u8;
            let e_eff_u8 = compute_pspm_eff_q7(
                eb as gcso_q7_t,
                explore_gain_u7,
                explore_pulse_q7,
                fold_geodesic,
            ) as u8;
            rotate_fused_ripa_and_pspm_uniform_subgroups(
                t_slice,
                b_u8,
                fact_h,
                f_eff_u8,
                logic_end,
                l_eff_u8,
                total_heads,
                e_eff_u8,
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
pub(super) fn dispatch_subhead_slot_ensemble_core(
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
                t_slice,
                logic_end,
                total_heads,
                head_dim,
                rot_dim,
                sin_e,
                cos_e,
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
            let raw_words = pack_q7_slice_to_swar_words_64(p_slice, total_heads);
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

