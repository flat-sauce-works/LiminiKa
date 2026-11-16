// name: src/core/src/traits/cold_path/storage_pprc.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Persona patching, L2P SVD projection, PPRC KV keyframes, DNP codecs, and mapped storage.

use super::*;

// ===================================================================
// 6. Persona Patch, L2P-SVD, KV-keyframe Keyframe Cache, DNP & mapped-storage Storage
// ===================================================================

/// Cold-path trait for dynamic persona phase modulation patches without altering base weights (**Macro Level**).
pub trait PersonaPatcher: BranchlessKernel + Send + Sync {
    /// Applies a binary persona phase modulation patch to the runtime context.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `patch_data` is empty or malformed.
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()>;

    /// Executes **Persistent Topological Mode Quantization** (Section 3.3) in row-major order,
    /// returning `Ok((0.0, 0.0))` with zeroed `profile_out` if total trail weight is zero.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `coherent_trails` is empty or `min_coherence_r_bar` is non-finite.
    #[inline]
    fn quantize_persistent_topological_mode(
        &self,
        coherent_trails: &[gcso_pointer_trail_t],
        min_coherence_r_bar: f32,
        profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(f32, f32)> {
        if coherent_trails.is_empty() || !branchless_is_finite_f32(min_coherence_r_bar) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let mut candidate = [0i8; MAX_ATTENTION_HEADS];
        let candidate_words: [u64; 8];
        if coherent_trails.len() == 1 {
            let trail = &coherent_trails[0];
            let d_bounded =
                branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
            let f_bounded =
                branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
            let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
            if w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            candidate = trail.accumulated_phase_delta;
            candidate_words = *trail_phase_words_ref(trail);
        } else {
            let mut sum_s = [[0.0_f32; 8]; 8];
            let mut sum_c = [[0.0_f32; 8]; 8];
            let mut word_s = [0.0_f32; 8];
            let mut word_c = [0.0_f32; 8];
            let mut total_w = 0.0_f32;
            let mut all_byte_uniform = true;
            let mut all_word_uniform = true;
            let mut uniform_s = 0.0_f32;
            let mut uniform_c = 0.0_f32;

            for trail in coherent_trails {
                let d_bounded =
                    branchless_clamp_f32(trail.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
                let f_bounded =
                    branchless_clamp_f32(trail.attractor_pull_force, 1.0, MAX_ATTRACTOR_PULL_FORCE);
                let w = branchless_flush_subnormal_f32(d_bounded * f_bounded);
                if w > 0.0 {
                    total_w += w;
                    let tw = *trail_phase_words_ref(trail);
                    let w0 = tw[0];
                    if is_word_array_uniform_64(&tw) {
                        if w0 == w0.rotate_right(8) {
                            let (s, c) = branchless_sincos_q7(w0 as gcso_q7_t);
                            uniform_s += w * s;
                            uniform_c += w * c;
                        } else {
                            all_byte_uniform = false;
                            let b = w0.to_le_bytes();
                            let mut lane = 0usize;
                            while lane < 8 {
                                let (s, c) = branchless_sincos_q7(b[lane] as gcso_q7_t);
                                word_s[lane] += w * s;
                                word_c[lane] += w * c;
                                lane += 1;
                            }
                        }
                    } else {
                        all_byte_uniform = false;
                        all_word_uniform = false;
                        let mut cell = 0usize;
                        while cell < 8 {
                            let b = tw[cell].to_le_bytes();
                            let mut lane = 0usize;
                            while lane < 8 {
                                let (s, c) = branchless_sincos_q7(b[lane] as gcso_q7_t);
                                sum_s[cell][lane] += w * s;
                                sum_c[cell][lane] += w * c;
                                lane += 1;
                            }
                            cell += 1;
                        }
                    }
                }
            }
            if total_w <= 1.0e-10 {
                *profile_out = [0i8; MAX_ATTENTION_HEADS];
                return Ok((0.0, 0.0));
            }
            if all_byte_uniform {
                let mag_sq = uniform_s * uniform_s + uniform_c * uniform_c;
                let has_mag = mag_sq > 1.0e-20;
                let q7 = branchless_select_q7(
                    has_mag,
                    branchless_atan2_to_q7_fast(uniform_s, uniform_c),
                    0,
                );
                let r_bar = branchless_select_f32(has_mag, 1.0, 0.0);
                let kappa_c = branchless_von_mises_kappa(r_bar);
                let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
                *profile_out = [branchless_select_q7(keep, q7, 0); MAX_ATTENTION_HEADS];
                return Ok((r_bar, kappa_c));
            } else if all_word_uniform {
                let mut s_cell = [0.0_f32; 8];
                let mut c_cell = [0.0_f32; 8];
                let mut lane = 0usize;
                while lane < 8 {
                    s_cell[lane] = word_s[lane] + uniform_s;
                    c_cell[lane] = word_c[lane] + uniform_c;
                    lane += 1;
                }
                let word0 = atan2_to_q7_word_x8_simd(&s_cell, &c_cell);
                candidate_words = [word0; 8];
                store_q7_words_64(&mut candidate, &candidate_words);
            } else {
                let has_word_or_uni = (uniform_s != 0.0)
                    | (uniform_c != 0.0)
                    | (word_s != [0.0_f32; 8])
                    | (word_c != [0.0_f32; 8]);
                if has_word_or_uni {
                    let mut cell = 0usize;
                    while cell < 8 {
                        let mut lane = 0usize;
                        while lane < 8 {
                            sum_s[cell][lane] += word_s[lane] + uniform_s;
                            sum_c[cell][lane] += word_c[lane] + uniform_c;
                            lane += 1;
                        }
                        cell += 1;
                    }
                }
                let mut cw = [0u64; 8];
                let mut cell = 0usize;
                while cell < 8 {
                    cw[cell] = atan2_to_q7_word_x8_simd(&sum_s[cell], &sum_c[cell]);
                    cell += 1;
                }
                candidate_words = cw;
                store_q7_words_64(&mut candidate, &candidate_words);
            }
        }

        let r_bar = circular_r_bar_from_words_64(&candidate_words);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let keep = r_bar >= branchless_clamp_f32(min_coherence_r_bar, 0.0, 1.0);
        if keep {
            *profile_out = candidate;
        } else {
            *profile_out = [0i8; MAX_ATTENTION_HEADS];
        }
        Ok((r_bar, kappa_c))
    }
}

/// Cold-path trait for Training-Free LoRA-to-Phase SVD Projection (L2P-SVD) (**Macro / Offline Level**).
pub trait L2pSvdProjector: BranchlessKernel + Send + Sync {
    /// Projects fine-tuned LoRA matrices ($W_A \in \mathbb{R}^{r \times d_{\mathrm{in}}}, W_B \in \mathbb{R}^{d_{\mathrm{out}} \times r}$)
    /// via first-order SVD approximation into phase profiles and Rank-1 Rank-1 residual vectors (Section 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `rank == 0`, `dim_in == 0`, `dim_out == 0`,
    /// `dim_out` is odd, matrix slices are smaller than `rank * dim_in` / `dim_out * rank`,
    /// or `phase_profile_out` is empty.
    #[inline]
    fn project_lora(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        rank1_out: &mut gcso_rank1_residual_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> GcsoResult<()> {
        if rank == 0
            || dim_in == 0
            || dim_out == 0
            || (dim_out & 1) != 0
            || lora_a.len()
                < rank
                    .checked_mul(dim_in)
                    .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?
            || lora_b.len()
                < dim_out
                    .checked_mul(rank)
                    .ok_or(GCSO_ERROR_INVALID_ARGUMENT)?
            || phase_profile_out.is_empty()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let b_slice = lora_b.as_slice();
        let a_slice = lora_a.as_slice();

        let norm_u_sq = if rank == 1 {
            sum_sq_f32_simd8(&b_slice[..dim_out])
        } else {
            let mut total_u = 0.0_f32;
            for i in 0..dim_out {
                let v = b_slice[i * rank];
                total_u += v * v;
            }
            total_u
        };
        let norm_v_sq = sum_sq_f32_simd8(&a_slice[..dim_in]);

        let sigma_1 = branchless_flush_subnormal_f32(branchless_sqrt_f32(norm_u_sq * norm_v_sq));
        rank1_out.layer_idx = 0;
        rank1_out.rank = 1;
        rank1_out.u_vector_ptr = lora_b.as_ptr() as u64;
        rank1_out.v_vector_ptr = lora_a.as_ptr() as u64;
        rank1_out.gain_scalar_ptr = 0;
        rank1_out.scale_factor = branchless_select_f32(sigma_1 > 0.0, sigma_1, 1.0);

        let out_phases = phase_profile_out.as_mut_slice();
        let num_heads = out_phases.len();
        let num_pairs = (dim_out / 2).max(1);
        let step_pairs = (num_pairs / num_heads.max(1)).max(1);
        let full_cells = num_heads >> 3;
        let mut cell = 0usize;
        while cell < full_cells {
            let h_base = cell << 3;
            let word = if rank == 1 && step_pairs == 1 && (h_base + 8) <= num_pairs {
                let base_f32 = h_base << 1;
                atan2_interleaved_pairs_to_q7_word_x8_simd(&b_slice[base_f32..base_f32 + 16])
            } else {
                let mut sin_arr = [0.0_f32; 8];
                let mut cos_arr = [0.0_f32; 8];
                let mut lane = 0usize;
                while lane < 8 {
                    let pair_idx = ((h_base + lane) * step_pairs) % num_pairs;
                    let base_idx = (pair_idx << 1) * rank;
                    cos_arr[lane] = b_slice[base_idx];
                    sin_arr[lane] = b_slice[base_idx + rank];
                    lane += 1;
                }
                atan2_to_q7_word_x8_simd(&sin_arr, &cos_arr)
            };
            let w_bytes = word.to_le_bytes();
            let mut lane_out = 0usize;
            while lane_out < 8 {
                out_phases[h_base + lane_out] = w_bytes[lane_out] as gcso_q7_t;
                lane_out += 1;
            }
            cell += 1;
        }
        let mut h = full_cells << 3;
        while h < num_heads {
            let pair_idx = (h * step_pairs) % num_pairs;
            let base_idx = (pair_idx << 1) * rank;
            out_phases[h] =
                branchless_atan2_to_q7_fast(b_slice[base_idx + rank], b_slice[base_idx]);
            h += 1;
        }

        Ok(())
    }

    /// Projects fine-tuned LoRA matrices into an Rank-1 residual Rank-1 descriptor, a Q7 rotational phase
    /// profile, and an **Isotropic Block-Diagonal Scaling** vector $\mathbf{D}$ (Theorem 1.1 & Section 5.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `rank == 0`, `dim_in == 0`, `dim_out == 0`,
    /// `dim_out` is odd, matrix slices are too small, `phase_profile_out` is empty, or
    /// `block_scales_out.len() < dim_out / 2`.
    #[inline]
    fn project_lora_with_block_scales(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        rank1_out: &mut gcso_rank1_residual_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        block_scales_out: &mut AlignedSliceMut32<'_, f32>,
    ) -> GcsoResult<()> {
        let num_pairs = dim_out / 2;
        if num_pairs == 0 || block_scales_out.len() < num_pairs {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.project_lora(
            lora_a,
            lora_b,
            rank,
            dim_in,
            dim_out,
            rank1_out,
            phase_profile_out,
        )?;

        let b_slice = lora_b.as_slice();
        let pair_stride = 2 * rank;
        let scales_slice = &mut block_scales_out.as_mut_slice()[..num_pairs];

        for (p_idx, scale_slot) in scales_slice.iter_mut().enumerate() {
            let row0 = p_idx * pair_stride;
            let pair_energy = sum_sq_f32_simd8(&b_slice[row0..row0 + pair_stride]);
            let mag = branchless_flush_subnormal_f32(branchless_sqrt_f32(pair_energy));
            *scale_slot = branchless_select_f32(mag > 1.0e-9, mag, 1.0);
        }
        rank1_out.gain_scalar_ptr = block_scales_out.as_ptr() as u64;
        Ok(())
    }
}

/// Mezzo/Cold-path trait for Predictive Phase-Motion & Residual Compensation (KV-keyframe) (**Mezzo / Storage Level**).
pub trait KvKeyframeCache: BranchlessKernel + Send + Sync {
    /// Seeks to `token_index` in the KV-keyframe keyframe cache and populates `header_out`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `token_index` is out of bounds.
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> GcsoResult<()>;

    /// Computes Circular Statistics resultant vector length $\bar{R}_c \in [0, 1]$ and von Mises
    /// concentration $\kappa_c$ using $\mathcal{O}(1)$ 256-entry Q7 lookup (`branchless_sincos_q7`)
    /// and division-free (`0 fdiv`) `branchless_inv_pos_f32` (Section 4.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `pruned_phases` is empty or its length does not
    /// match `token_weights.len()`.
    #[inline]
    fn aggregate_pruned_kv_circular_stats(
        &self,
        pruned_phases: &AlignedSlice16<'_, gcso_q7_t>,
        token_weights: &AlignedSlice32<'_, f32>,
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> GcsoResult<(f32, f32)> {
        if pruned_phases.is_empty() || pruned_phases.len() != token_weights.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let phases = pruned_phases.as_slice();
        let weights = token_weights.as_slice();
        let n = phases.len();
        let mut sum_c0 = 0.0_f32;
        let mut sum_s0 = 0.0_f32;
        let mut sum_w0 = 0.0_f32;
        let mut sum_c1 = 0.0_f32;
        let mut sum_s1 = 0.0_f32;
        let mut sum_w1 = 0.0_f32;

        let mut idx = 0usize;
        while idx + 2 <= n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;

            let w1 = branchless_relu_f32(weights[idx + 1]);
            let (s1, c1) = branchless_sincos_q7(phases[idx + 1]);
            sum_c1 += w1 * c1;
            sum_s1 += w1 * s1;
            sum_w1 += w1;

            idx += 2;
        }
        if idx < n {
            let w0 = branchless_relu_f32(weights[idx]);
            let (s0, c0) = branchless_sincos_q7(phases[idx]);
            sum_c0 += w0 * c0;
            sum_s0 += w0 * s0;
            sum_w0 += w0;
        }
        let sum_c = sum_c0 + sum_c1;
        let sum_s = sum_s0 + sum_s1;
        let sum_w = sum_w0 + sum_w1;

        let norm = branchless_sqrt_f32(sum_c * sum_c + sum_s * sum_s);
        let r_bar = branchless_flush_subnormal_f32(branchless_select_f32(
            sum_w > DEFAULT_ENTROPY_EPS,
            norm * branchless_inv_pos_f32(sum_w),
            0.0,
        ));
        let r_bar_clamped = branchless_clamp_f32(r_bar, 0.0, 1.0);
        let kappa_c = branchless_von_mises_kappa(r_bar_clamped);
        let theta_bar = branchless_atan2_f32(sum_s, sum_c);

        header_out.composite_vector_length = r_bar_clamped;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = theta_bar;
        Ok((r_bar_clamped, kappa_c))
    }

    /// Computes Multi-Head Virtual Position Shift $\Delta m^{(h)}$ in division-free (`0 fdiv`)
    /// branchless mode (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn compute_virtual_position_shift(
        &self,
        accumulated_phase_rad: f32,
        mean_base_rope_freq: f32,
    ) -> f32 {
        let abs_freq = branchless_abs_f32(mean_base_rope_freq);
        let valid_freq = (abs_freq > 1.0e-9) & branchless_is_finite_f32(mean_base_rope_freq);
        let inv_freq = branchless_copysign_f32(
            branchless_inv_pos_f32(branchless_select_f32(valid_freq, abs_freq, 1.0)),
            mean_base_rope_freq,
        );
        branchless_select_f32(
            valid_freq,
            branchless_flush_subnormal_f32(accumulated_phase_rad * inv_freq),
            0.0,
        )
    }

    /// Evaluates the **Variable Group of Pictures (GOP) Structure** keyframe classification
    /// (`0` = I-Frame Keyframe Anchor, `1` = P-Frame Phase Motion Delta) and effective GOP length (Section 4.3).
    #[inline(always)]
    #[must_use]
    fn evaluate_variable_gop_trigger(
        &self,
        moving_z_entropy: f32,
        sliding_flux: f32,
        steps_since_keyframe: u32,
        min_gop: u32,
        max_gop: u32,
        entropy_surge_thresh: f32,
        flux_surge_thresh: f32,
    ) -> (u32, u32) {
        let safe_min = branchless_select_u32(min_gop > 0, min_gop, 1);
        let safe_max = branchless_select_u32(max_gop >= safe_min, max_gop, safe_min);
        let clean_h = branchless_select_f32(
            branchless_is_finite_f32(moving_z_entropy),
            moving_z_entropy,
            0.0,
        );
        let clean_flux =
            branchless_select_f32(branchless_is_finite_f32(sliding_flux), sliding_flux, 0.0);

        let surge = (clean_h > entropy_surge_thresh) | (clean_flux > flux_surge_thresh);
        let is_iframe =
            (steps_since_keyframe >= safe_max) | ((steps_since_keyframe >= safe_min) & surge);
        let frame_type = branchless_select_u32(is_iframe, 0, 1);
        let gop_len = branchless_select_u32(is_iframe, steps_since_keyframe.max(1), safe_max);
        (frame_type, gop_len)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Replay)** phase reconstruction (Section 4.3)
    /// using 8-lane SWAR saturating addition and SWAR QDPS filtering without forward passes.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_phase_trajectory(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if qdps_min_step_q7 > 127 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        *reconstructed_out = *icache_anchor_phases;
        for delta in pcache_motion_deltas {
            branchless_q7_accumulate_64(reconstructed_out, delta);
        }
        branchless_q7_qdps_filter_64(reconstructed_out, qdps_min_step_q7);
        Ok(())
    }

    /// Executes **Seam-Free $S^1$ Geodesic P-Frame Trajectory Replay** (Section 4.3):
    /// accumulates P-Frame motion deltas on the circle group $S^1 \cong \mathbb{Z}/256\mathbb{Z}$,
    /// folds onto `[-64, +64]`, and applies 8-lane SWAR QDPS step filtering.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127`.
    #[inline]
    fn reconstruct_pframe_geodesic_trajectory(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<()> {
        if qdps_min_step_q7 > 127 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            cur_w[0] = swar_q7_circular_add_x8(cur_w[0], dw[0]);
            cur_w[1] = swar_q7_circular_add_x8(cur_w[1], dw[1]);
            cur_w[2] = swar_q7_circular_add_x8(cur_w[2], dw[2]);
            cur_w[3] = swar_q7_circular_add_x8(cur_w[3], dw[3]);
            cur_w[4] = swar_q7_circular_add_x8(cur_w[4], dw[4]);
            cur_w[5] = swar_q7_circular_add_x8(cur_w[5], dw[5]);
            cur_w[6] = swar_q7_circular_add_x8(cur_w[6], dw[6]);
            cur_w[7] = swar_q7_circular_add_x8(cur_w[7], dw[7]);
        }
        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_qdps_filter_words_64(&folded_w, qdps_min_step_q7);
        store_q7_words_64(reconstructed_out, &filt_w);
        Ok(())
    }

    /// Replays a **Limina P-Frame Motion Chain** ($\boldsymbol{\Delta\theta}_1, \dots, \boldsymbol{\Delta\theta}_T$)
    /// from an **I-Frame Key Anchor** (`icache_anchor_phases`), evaluating the **Joint Probability
    /// Chain Rule Log-Likelihood** $\ln P(M_1, \dots, M_T) = \sum_{t=1}^T \ln P(M_t \mid M_{t-1})$
    /// and inverting any self-contradictory ($H^1(K; \mathcal{F}) \neq 0$) P-Frame step via
    /// Phase-Conjugate Repulsion ($-\boldsymbol{\Delta\theta}_t$).
    ///
    /// Populates `header_out` (`gop_length`, `composite_vector_length = r_bar`,
    /// `von_mises_kappa = kappa_c`, `sparse_scalar_residual = joint_log_prob`) and returns
    /// `(joint_log_prob, r_bar, kappa_c, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127` or `contradiction_threshold` is non-finite.
    #[inline]
    fn replay_liminal_pframe_chain_with_joint_log_prob(
        &self,
        icache_anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        pcache_motion_deltas: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        reconstructed_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
        header_out: &mut gcso_kv_keyframe_header_t,
    ) -> GcsoResult<(f32, f32, f32, u32)> {
        if qdps_min_step_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(icache_anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for delta in pcache_motion_deltas {
            let dw = load_q7_words_64(delta);
            // Because `next_w = cur_w (+) dw` in Z_256^64 and cos(-d) = cos(d), the transition
            // coherence C(cur_w, next_w) depends purely on `dw` against zero, avoiding a
            // tentative 8-word addition + 8-word subtraction before checking repulsion.
            let coh = coherence_from_diff_words_64(
                dw[0], dw[1], dw[2], dw[3], dw[4], dw[5], dw[6], dw[7],
            );
            let log_p = branchless_fast_transition_log_prob(coh);
            if coh < -safe_thresh {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], swar_q7_neg_sat_x8(dw[0])),
                    swar_q7_circular_add_x8(cur_w[1], swar_q7_neg_sat_x8(dw[1])),
                    swar_q7_circular_add_x8(cur_w[2], swar_q7_neg_sat_x8(dw[2])),
                    swar_q7_circular_add_x8(cur_w[3], swar_q7_neg_sat_x8(dw[3])),
                    swar_q7_circular_add_x8(cur_w[4], swar_q7_neg_sat_x8(dw[4])),
                    swar_q7_circular_add_x8(cur_w[5], swar_q7_neg_sat_x8(dw[5])),
                    swar_q7_circular_add_x8(cur_w[6], swar_q7_neg_sat_x8(dw[6])),
                    swar_q7_circular_add_x8(cur_w[7], swar_q7_neg_sat_x8(dw[7])),
                ];
                repulsed_count = repulsed_count.wrapping_add(1);
            } else {
                cur_w = [
                    swar_q7_circular_add_x8(cur_w[0], dw[0]),
                    swar_q7_circular_add_x8(cur_w[1], dw[1]),
                    swar_q7_circular_add_x8(cur_w[2], dw[2]),
                    swar_q7_circular_add_x8(cur_w[3], dw[3]),
                    swar_q7_circular_add_x8(cur_w[4], dw[4]),
                    swar_q7_circular_add_x8(cur_w[5], dw[5]),
                    swar_q7_circular_add_x8(cur_w[6], dw[6]),
                    swar_q7_circular_add_x8(cur_w[7], dw[7]),
                ];
            }
            joint_log_prob += log_p;
        }

        let folded_w = [
            swar_q7_fold_s1_geodesic_x8(cur_w[0]),
            swar_q7_fold_s1_geodesic_x8(cur_w[1]),
            swar_q7_fold_s1_geodesic_x8(cur_w[2]),
            swar_q7_fold_s1_geodesic_x8(cur_w[3]),
            swar_q7_fold_s1_geodesic_x8(cur_w[4]),
            swar_q7_fold_s1_geodesic_x8(cur_w[5]),
            swar_q7_fold_s1_geodesic_x8(cur_w[6]),
            swar_q7_fold_s1_geodesic_x8(cur_w[7]),
        ];
        let filt_w = swar_q7_qdps_filter_words_64(&folded_w, qdps_min_step_q7);
        store_q7_words_64(reconstructed_out, &filt_w);

        let r_bar = circular_r_bar_from_words_64(&filt_w);
        let kappa_c = branchless_von_mises_kappa(r_bar);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        header_out.gop_length = (pcache_motion_deltas.len() as u32).max(1);
        header_out.composite_vector_length = r_bar;
        header_out.von_mises_kappa = kappa_c;
        header_out.sparse_scalar_residual = clean_lp;

        Ok((clean_lp, r_bar, kappa_c, repulsed_count))
    }

    /// Encodes a 128-byte [`KvKeyframePacket`] (**I-Frame Key Anchor**, Section 4.3) from a
    /// 64-head Q7 phase profile (`anchor_phases`), computing Circular Statistics resultant vector
    /// length $\bar{R}_c \in [0, 1]$, von Mises concentration $\kappa_c$, resultant mean phase
    /// angle $\bar{\theta}_c$, and stamping the IEEE 802.3 CRC32 integrity checksum plus causal
    /// provenance metadata (`provenance_bits`, `cluster_id`, `slot_id`).
    #[inline]
    #[must_use]
    fn encode_pprc_iframe_packet(
        &self,
        token_index: u32,
        gop_length: u32,
        anchor_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        provenance_bits: u32,
        cluster_id: u32,
        slot_id: u8,
    ) -> KvKeyframePacket {
        let words = load_q7_words_64(anchor_phases);
        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&words);
        let theta_bar = branchless_flush_subnormal_f32((mean_q7 as f32) * Q7_PHASE_SCALE_RAD);
        KvKeyframePacket::new(
            token_index,
            gop_length.max(1),
            r_bar,
            kappa_c,
            theta_bar,
            provenance_bits,
            cluster_id,
            slot_id,
            *anchor_phases,
        )
    }

    /// Encodes a 64-byte [`KvMotionDeltaPacket`] (**P-Frame Predictive Phase-Motion Delta**
    /// $\boldsymbol{\Delta\theta}_t$, Section 4.3) between `prev_phases` ($\boldsymbol{\theta}_{t-1}$)
    /// and `curr_phases` ($\boldsymbol{\theta}_t$) using 8-lane SWAR circular difference
    /// (`swar_q7_circular_diff_x8`), $S^1$ geodesic folding (`swar_q7_fold_s1_geodesic_x8`), and
    /// QDPS step filtering (`swar_q7_qdps_filter_words_64`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdps_min_step_q7 > 127`.
    #[inline]
    fn encode_pprc_pframe_delta_packet(
        &self,
        prev_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        curr_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        qdps_min_step_q7: u8,
    ) -> GcsoResult<KvMotionDeltaPacket> {
        encode_pframe_motion_delta_64(prev_phases, curr_phases, qdps_min_step_q7)
    }

    /// Executes **Zero-Forward Latency Seek (Instantaneous Persona & Memory Replay)** over a
    /// lightweight KV-keyframe Variable GOP stream (`keyframe` I-Frame + `pframes[..seek_pframe_steps]`
    /// P-Frame deltas), reconstructing the historical 64-head persona & episodic memory state
    /// directly onto `trail_out` without running any LLM forward pass (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if `seek_pframe_steps > pframes.len()`,
    /// `qdps_min_step_q7 > 127`, or `contradiction_threshold` is non-finite.
    #[inline]
    fn seek_and_reconstruct_persona_from_gop(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        seek_pframe_steps: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, f32, f32, u32)> {
        keyframe.verify_integrity()?;
        let (filt_w, clean_lp, repulsed_count) = seek_pframe_chain_words_64(
            &keyframe.anchor_phases,
            pframes,
            seek_pframe_steps,
            qdps_min_step_q7,
            contradiction_threshold,
        )?;
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(gcso_tagged_ptr_t(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe.header.token_index.wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = gcso_tagged_ptr_t::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.provenance_bits());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }

    /// Serializes a lightweight KV-keyframe Variable GOP stream (`1` 128-byte [`KvKeyframePacket`] +
    /// `pframes.len()` 64-byte [`KvMotionDeltaPacket`]s) into `dst_bytes` (`128 + N * 64` bytes),
    /// returning the total byte length written.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `keyframe.verify_integrity()` fails, or
    /// [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < 128 + pframes.len() * 64`.
    #[inline]
    fn pack_kv_gop_stream_bytes(
        &self,
        keyframe: &KvKeyframePacket,
        pframes: &[KvMotionDeltaPacket],
        dst_bytes: &mut [u8],
    ) -> GcsoResult<usize> {
        keyframe.verify_integrity()?;
        let p_bytes = pframes
            .len()
            .checked_mul(core::mem::size_of::<KvMotionDeltaPacket>())
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let total_bytes = core::mem::size_of::<KvKeyframePacket>()
            .checked_add(p_bytes)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if dst_bytes.len() < total_bytes {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        keyframe.write_unaligned_to(dst_bytes, 0)?;
        KvMotionDeltaPacket::write_unaligned_slice_to(
            pframes,
            dst_bytes,
            core::mem::size_of::<KvKeyframePacket>(),
        )?;
        Ok(total_bytes)
    }

    /// Unpacks a serialized KV-keyframe Variable GOP byte stream (`stream_bytes`) in $\mathcal{O}(1)$
    /// stack space ($C_{\mathrm{alloc}} = 0$) and executes **Zero-Forward Latency Seek** up to
    /// `seek_pframe_steps`, reconstructing the historical persona and episodic memory state onto
    /// `trail_out` (Section 4.3).
    ///
    /// Returns `(joint_log_prob, r_bar, kappa_c, virtual_pos_shift, repulsed_pframe_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `stream_bytes` is truncated,
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] if the I-Frame header or CRC32 is invalid, or
    /// [`GCSO_ERROR_INVALID_ARGUMENT`] if parameters are out of range.
    #[inline]
    fn unpack_and_seek_kv_gop_stream_bytes(
        &self,
        stream_bytes: &[u8],
        seek_pframe_steps: usize,
        qdps_min_step_q7: u8,
        contradiction_threshold: f32,
        mean_base_rope_freq: f32,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(f32, f32, f32, f32, u32)> {
        if qdps_min_step_q7 > 127 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let iframe_size = core::mem::size_of::<KvKeyframePacket>();
        let pframe_size = core::mem::size_of::<KvMotionDeltaPacket>();
        let needed_bytes = seek_pframe_steps
            .checked_mul(pframe_size)
            .and_then(|b| b.checked_add(iframe_size))
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if stream_bytes.len() < needed_bytes {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        let keyframe = KvKeyframePacket::read_unaligned_from(stream_bytes, 0)?;
        keyframe.verify_integrity()?;

        let safe_thresh =
            branchless_clamp_f32(branchless_abs_f32(contradiction_threshold), 0.0, 1.0);
        let mut cur_w = load_q7_words_64(&keyframe.anchor_phases);
        let mut joint_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        let mut step = 0usize;
        while step < seek_pframe_steps {
            let off = iframe_size + step * pframe_size;
            let pframe = KvMotionDeltaPacket::read_unaligned_from(stream_bytes, off)?;
            let (log_p, repulsed) =
                apply_pframe_motion_step_words_64(&mut cur_w, &pframe, safe_thresh);
            joint_log_prob += log_p;
            if repulsed {
                repulsed_count = repulsed_count.wrapping_add(1);
            }
            step += 1;
        }

        let filt_w = finalize_pframe_chain_words_64(&cur_w, qdps_min_step_q7);
        store_q7_words_64(&mut trail_out.accumulated_phase_delta, &filt_w);

        let (r_bar, mean_q7, kappa_c) = circular_resultant_from_words_64(&filt_w);
        let clean_lp = branchless_flush_subnormal_f32(joint_log_prob);
        let mean_rad = (mean_q7 as f32) * Q7_PHASE_SCALE_RAD;
        let virtual_pos_shift = self.compute_virtual_position_shift(mean_rad, mean_base_rope_freq);

        let cur_mode = unpack_action_mode_from_tag(gcso_tagged_ptr_t(trail_out.current_ptr).unpack_tag());
        let target_token = keyframe.header.token_index.wrapping_add(seek_pframe_steps as u32);
        trail_out.current_ptr = gcso_tagged_ptr_t::pack_action(
            u64::from(target_token),
            keyframe.slot_id(),
            cur_mode,
            mean_q7,
        )
        .raw();
        trail_out.user_data |= u64::from(keyframe.provenance_bits());
        trail_out.target_anchor_id = keyframe.cluster_id();
        trail_out.cluster_id = keyframe.cluster_id();
        if trail_out.stigmergic_density < r_bar {
            trail_out.stigmergic_density = r_bar;
        }
        Ok((clean_lp, r_bar, kappa_c, virtual_pos_shift, repulsed_count))
    }
}

/// Trait for Dynamic Node Protocol (GCSO-DNP) distributed phase synchronization (**FastPathLedger Mode 1 / Boundary Layer**).
pub trait DnpPacketDispatcher: BranchlessKernel + Send + Sync {
    /// Dispatches a compact 64-byte GCSO-DNP phase/repulsion synchronization packet in zero-allocation mode.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `ledger_slot.mode != 1` (Network Plugin Mode).
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
    #[inline]
    fn dispatch_dnp_packet(
        &mut self,
        ledger_slot: &gcso_ledger_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
    ) -> GcsoResult<()> {
        if ledger_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_payload.len() < 64 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }

    /// Encodes a 64-head Q7 phase/repulsion vector into a 64-byte cacheline-aligned GCSO-DNP
    /// packet (`packet_out`) using bulk `Pod` copy and returns its branchless IEEE 802.3 CRC32 integrity tag.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `ledger_slot.mode != 1`.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_out.len() < 64` or `source_slot_id >= 8`.
    #[inline]
    fn encode_dnp_phase_packet(
        &self,
        ledger_slot: &gcso_ledger_slot_t,
        source_slot_id: u8,
        action_mode_flags: u8,
        phase_deltas: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        packet_out: &mut AlignedSliceMut64<'_, u8>,
    ) -> GcsoResult<u32> {
        if ledger_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_out.len() < MAX_ATTENTION_HEADS
            || (source_slot_id as usize) >= MAX_ATTRACTOR_DOMAINS
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let phase_w = load_q7_words_64(phase_deltas);
        let dst = &mut packet_out.as_mut_slice()[..MAX_ATTENTION_HEADS];
        let mut cell = 0usize;
        while cell < 8 {
            let base = cell << 3;
            dst[base..base + 8].copy_from_slice(&phase_w[cell].to_le_bytes());
            cell += 1;
        }
        let tag_byte = pack_action_slot_tag(source_slot_id, action_mode_flags);
        let crc = crc32_block64_zero_seed_words(&phase_w) ^ u32::from(tag_byte);
        Ok(crc)
    }

    /// Verifies and decodes a 64-byte cacheline-aligned GCSO-DNP packet (`packet_payload`) into
    /// 8 register-resident `u64` SWAR words and the recovered 8-bit tag byte.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `ledger_slot.mode != 1`.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
    /// checksum do not match `expected_crc32`.
    #[inline(always)]
    fn decode_and_verify_dnp_words(
        &self,
        ledger_slot: &gcso_ledger_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
        expected_crc32: u32,
    ) -> GcsoResult<([u64; 8], u8)> {
        if ledger_slot.mode != 1 {
            return Err(GCSO_ERROR_INVALID_STATE);
        }
        if packet_payload.len() < MAX_ATTENTION_HEADS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let src = &packet_payload.as_slice()[..MAX_ATTENTION_HEADS];
        let payload_words = [
            u64::from_le_bytes([
                src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
            ]),
            u64::from_le_bytes([
                src[8], src[9], src[10], src[11], src[12], src[13], src[14], src[15],
            ]),
            u64::from_le_bytes([
                src[16], src[17], src[18], src[19], src[20], src[21], src[22], src[23],
            ]),
            u64::from_le_bytes([
                src[24], src[25], src[26], src[27], src[28], src[29], src[30], src[31],
            ]),
            u64::from_le_bytes([
                src[32], src[33], src[34], src[35], src[36], src[37], src[38], src[39],
            ]),
            u64::from_le_bytes([
                src[40], src[41], src[42], src[43], src[44], src[45], src[46], src[47],
            ]),
            u64::from_le_bytes([
                src[48], src[49], src[50], src[51], src[52], src[53], src[54], src[55],
            ]),
            u64::from_le_bytes([
                src[56], src[57], src[58], src[59], src[60], src[61], src[62], src[63],
            ]),
        ];
        let raw_crc = crc32_block64_zero_seed_words(&payload_words);
        let diff = raw_crc ^ expected_crc32;
        if (diff & 0xFFFF_FF00) != 0 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok((payload_words, (diff & 0xFF) as u8))
    }

    /// Verifies and decodes a 64-byte cacheline-aligned GCSO-DNP packet (`packet_payload`) into
    /// a 64-head Q7 phase/repulsion vector (`phases_out`), returning the recovered 8-bit tag byte.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_STATE`] if `ledger_slot.mode != 1`.
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `packet_payload.len() < 64`.
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if the upper 24 bits of the IEEE 802.3 CRC32
    /// checksum do not match `expected_crc32`.
    #[inline]
    fn decode_and_verify_dnp_packet(
        &self,
        ledger_slot: &gcso_ledger_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
        expected_crc32: u32,
        phases_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<u8> {
        match self.decode_and_verify_dnp_words(ledger_slot, packet_payload, expected_crc32) {
            Ok((payload_words, tag_byte)) => {
                store_q7_words_64(phases_out, &payload_words);
                Ok(tag_byte)
            }
            Err(err) => {
                if err == GCSO_ERROR_CONTAINER_CORRUPTED {
                    *phases_out = [0i8; MAX_ATTENTION_HEADS];
                }
                Err(err)
            }
        }
    }

    /// Verifies a 64-byte GCSO-DNP packet and applies its decoded 64-head phase/repulsion vector
    /// directly to `trail` (and `bloom_filter` if repulsive).
    ///
    /// # Errors
    /// Propagates any error returned by [`Self::decode_and_verify_dnp_packet`].
    #[inline]
    fn decode_and_apply_dnp_packet_to_trail(
        &self,
        ledger_slot: &gcso_ledger_slot_t,
        packet_payload: &AlignedSlice64<'_, u8>,
        expected_crc32: u32,
        inhibition_strength: f32,
        bloom_filter: Option<&mut ContrastivePhaseBloomFilter>,
        trail: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<(u8, u8, bool)> {
        let (payload_words, tag_byte) =
            self.decode_and_verify_dnp_words(ledger_slot, packet_payload, expected_crc32)?;
        let slot_id = unpack_attractor_slot_from_tag(tag_byte);
        let mode_flags = unpack_action_mode_from_tag(tag_byte);
        let is_repulsion =
            (slot_id == ATTRACTOR_SLOT_REPULSION) | ((mode_flags & ACTION_MODE_PHASE_REPULSE) != 0);

        if is_repulsion {
            if let Some(bf) = bloom_filter {
                bf.insert_repulsive_pattern_words_weighted(&payload_words, 1);
            }
            let cur_pull = trail.attractor_pull_force;
            let damped = branchless_q7_inhibit_words_64(
                trail_phase_words_mut(trail),
                cur_pull,
                &payload_words,
                inhibition_strength,
            );
            trail.attractor_pull_force = branchless_min_f32(
                branchless_flush_subnormal_f32(damped),
                MAX_ATTRACTOR_PULL_FORCE,
            );
            trail.flags |= TRAIL_FLAG_REPULSION_ACTIVE;
        } else {
            let tw = trail_phase_words_mut(trail);
            tw[0] = swar_q7_add_sat_x8(tw[0], payload_words[0]);
            tw[1] = swar_q7_add_sat_x8(tw[1], payload_words[1]);
            tw[2] = swar_q7_add_sat_x8(tw[2], payload_words[2]);
            tw[3] = swar_q7_add_sat_x8(tw[3], payload_words[3]);
            tw[4] = swar_q7_add_sat_x8(tw[4], payload_words[4]);
            tw[5] = swar_q7_add_sat_x8(tw[5], payload_words[5]);
            tw[6] = swar_q7_add_sat_x8(tw[6], payload_words[6]);
            tw[7] = swar_q7_add_sat_x8(tw[7], payload_words[7]);
        }
        Ok((slot_id, mode_flags, is_repulsion))
    }
}

/// Cold-path trait for Zero-Overhead In-Memory Mapped Storage (mapped-storage) (**Storage / Persistence Level**).
pub trait MappedStorage: BranchlessKernel + Send + Sync {
    /// Serializes active runtime state into a `.gcso` binary snapshot buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `buffer.len() < *required_size`.
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> GcsoResult<()>;

    /// Deserializes a `.gcso` binary snapshot buffer to restore context state.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if the snapshot header or checksum is invalid.
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> GcsoResult<()>;

    /// Opens a memory-mapped zero-copy handle for a `.gcso` container payload.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `file_path` is empty, or
    /// [`crate::abi::GCSO_ERROR_MAPPED_STORAGE_FAILED`] if OS `mmap` fails.
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_mapped_storage_descriptor_t,
    ) -> GcsoResult<()>;

    /// Unmaps a zero-copy mapped-storage memory handle and releases Direct-DMA resources.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_IO_FAILURE`] if unmapping fails.
    fn close_mmap(&mut self, descriptor: &mut gcso_mapped_storage_descriptor_t) -> GcsoResult<()>;

    /// Retrieves the byte offset for a specific track in a `.gcso` container
    /// (`0` = `CORE`, `1` = `I-CACHE`, `2` = `P-CACHE`, `3` = `SNAPSHOT`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3`.
    #[inline(always)]
    fn get_track_offset(&self, track_id: u32) -> GcsoResult<u64> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>() as u64;
        match track_id {
            0 => Ok(header_size),
            1 => Ok(header_size + 1024),
            2 => Ok(header_size + 2048),
            3 => Ok(header_size + 3072),
            _ => Err(GCSO_ERROR_INVALID_ARGUMENT),
        }
    }

    /// Reads and validates a `.gcso` container's 128-byte [`gcso_snapshot_header_t`] via [`verify_snapshot_payload_crc32`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if `container_bytes` is truncated, magic/offsets
    /// are invalid, or CRC32 verification fails.
    /// Returns [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`] if `version != 0x00000101`.
    #[inline]
    fn inspect_and_verify_container(
        &self,
        container_bytes: &[u8],
    ) -> GcsoResult<gcso_snapshot_header_t> {
        let header = gcso_snapshot_header_t::read_unaligned_from(container_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;
        verify_snapshot_payload_crc32(&header, container_bytes)?;
        Ok(header)
    }

    /// Finalizes the IEEE 802.3 CRC32 payload checksum on `header` and writes the 128-byte header
    /// into the beginning of `container_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`crate::abi::GCSO_ERROR_VERSION_MISMATCH`]
    /// if `header` or `container_bytes` bounds are invalid.
    #[inline]
    fn stamp_and_write_snapshot_header(
        &self,
        header: &mut gcso_snapshot_header_t,
        container_bytes: &mut [u8],
    ) -> GcsoResult<u32> {
        let crc = finalize_snapshot_header_crc32(header, container_bytes)?;
        header
            .write_unaligned_to(container_bytes, 0)
            .map_err(|_| GCSO_ERROR_CONTAINER_CORRUPTED)?;
        Ok(crc)
    }

    /// Extracts a zero-copy byte slice `&'a [u8]` of length `track_len` for `track_id` (`0..=3`) from `container_bytes`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3`.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `container_bytes` cannot hold `offset + track_len`.
    #[inline]
    fn extract_track_slice<'a>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        track_len: usize,
    ) -> GcsoResult<&'a [u8]> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let end = start
            .checked_add(track_len)
            .ok_or(GCSO_ERROR_BUFFER_TOO_SMALL)?;
        if container_bytes.len() < end {
            return Err(GCSO_ERROR_BUFFER_TOO_SMALL);
        }
        Ok(&container_bytes[start..end])
    }

    /// Extracts a hardware-aligned zero-copy [`AlignedSlice`] (`AlignedSlice<'a, T, ALIGN>`) of `elem_count` elements
    /// for `track_id` (`0..=3`) directly from a mapped-storage memory-mapped `.gcso` container buffer.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `track_id > 3` or `T` is zero-sized.
    /// Returns [`GCSO_ERROR_BUFFER_TOO_SMALL`] if `container_bytes` is too small.
    /// Returns [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if the track start address violates `ALIGN`.
    #[inline]
    fn extract_aligned_track_slice<'a, T: Pod, const ALIGN: usize>(
        &self,
        container_bytes: &'a [u8],
        track_id: u32,
        elem_count: usize,
    ) -> GcsoResult<AlignedSlice<'a, T, ALIGN>> {
        let start = usize::try_from(self.get_track_offset(track_id)?)
            .map_err(|_| GCSO_ERROR_BUFFER_TOO_SMALL)?;
        let typed_slice = T::slice_from_bytes_at(container_bytes, start, elem_count)?;
        AlignedSlice::<T, ALIGN>::new(typed_slice)
    }
}

