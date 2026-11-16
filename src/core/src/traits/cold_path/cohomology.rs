// name: src/core/src/traits/cold_path/cohomology.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cellular sheaf cohomology, Chebyshev spectral filtering, and dual pairing traits.

use super::*;

// ===================================================================
// 1. Cellular Sheaf Cohomology & Matrix-Free Spectral Filtering
// ===================================================================

/// Cold-path trait for Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ evaluation, matrix-free
/// spectral filtering, and Joint Probability Chain Rule / Limina trail-chain verification (**Macro Level**).
pub trait SheafCohomologyEvaluator: BranchlessKernel + Send + Sync {
    /// Evaluates the Gershgorin upper bound $\hat{\lambda}_{\max} = 2 \cdot \max_i D_{ii}$ in $\mathcal{O}(\vert V \vert)$ time (Section 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `node_degrees` is empty.
    #[inline]
    fn eval_gershgorin_lambda_max(
        &self,
        node_degrees: &AlignedSlice32<'_, f32>,
    ) -> GcsoResult<f32> {
        if node_degrees.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(branchless_flush_subnormal_f32(
            2.0 * max_pos_f32_unrolled4(node_degrees.as_slice()),
        ))
    }

    /// Evaluates coarse supernode degrees $D_{ii} = \sum_{j \neq i} A_{ij}$ and returns
    /// $\hat{\lambda}_{\max} = 2 \max_i D_{ii}$ directly from 256-bit PagedBlock bitmasks (Section 1.2 & Lemma 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `block_masks` is empty or `degrees_out.len() < block_masks.len()`.
    #[inline]
    fn eval_stigmergic_block_degrees(
        &self,
        block_masks: &[gcso_paged_bitmask_t],
        degrees_out: &mut AlignedSliceMut32<'_, f32>,
    ) -> GcsoResult<f32> {
        let n = block_masks.len();
        if n == 0 || degrees_out.len() < n {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let deg_slice = &mut degrees_out.as_mut_slice()[..n];
        deg_slice.fill(0.0);

        let mut inv_norm_cache = [0.0_f32; BLOCK_NORM_CACHE_CAP];
        let cached_len = n.min(BLOCK_NORM_CACHE_CAP);
        for (slot, mask) in inv_norm_cache[..cached_len]
            .iter_mut()
            .zip(block_masks[..cached_len].iter())
        {
            *slot = branchless_popcount_rsqrt_u32(mask.popcount());
        }

        let mut max_deg = 0.0_f32;
        let cached_upper = cached_len;
        for i in 0..n {
            let mask_i = &block_masks[i];
            let inv_norm_i = if i < BLOCK_NORM_CACHE_CAP {
                inv_norm_cache[i]
            } else {
                branchless_popcount_rsqrt_u32(mask_i.popcount())
            };

            if inv_norm_i > 0.0 {
                let [m0, m1, m2, m3] = mask_i.bits;
                let mut acc_i0 = deg_slice[i];
                let mut acc_i1 = 0.0_f32;
                let mut j = i + 1;

                while j + 4 <= cached_upper {
                    let [w0_a, w1_a, w2_a, w3_a] = block_masks[j].bits;
                    let [w0_b, w1_b, w2_b, w3_b] = block_masks[j + 1].bits;
                    let [w0_c, w1_c, w2_c, w3_c] = block_masks[j + 2].bits;
                    let [w0_d, w1_d, w2_d, w3_d] = block_masks[j + 3].bits;

                    let inter_a = ((m0 & w0_a).count_ones() + (m1 & w1_a).count_ones())
                        + ((m2 & w2_a).count_ones() + (m3 & w3_a).count_ones());
                    let inter_b = ((m0 & w0_b).count_ones() + (m1 & w1_b).count_ones())
                        + ((m2 & w2_b).count_ones() + (m3 & w3_b).count_ones());
                    let inter_c = ((m0 & w0_c).count_ones() + (m1 & w1_c).count_ones())
                        + ((m2 & w2_c).count_ones() + (m3 & w3_c).count_ones());
                    let inter_d = ((m0 & w0_d).count_ones() + (m1 & w1_d).count_ones())
                        + ((m2 & w2_d).count_ones() + (m3 & w3_d).count_ones());

                    let a_ij_a = (inter_a as f32) * (inv_norm_i * inv_norm_cache[j]);
                    let a_ij_b = (inter_b as f32) * (inv_norm_i * inv_norm_cache[j + 1]);
                    let a_ij_c = (inter_c as f32) * (inv_norm_i * inv_norm_cache[j + 2]);
                    let a_ij_d = (inter_d as f32) * (inv_norm_i * inv_norm_cache[j + 3]);

                    deg_slice[j] += a_ij_a;
                    deg_slice[j + 1] += a_ij_b;
                    deg_slice[j + 2] += a_ij_c;
                    deg_slice[j + 3] += a_ij_d;
                    acc_i0 += a_ij_a + a_ij_b;
                    acc_i1 += a_ij_c + a_ij_d;
                    j += 4;
                }
                while j < cached_upper {
                    let [w0, w1, w2, w3] = block_masks[j].bits;
                    let inter = ((m0 & w0).count_ones() + (m1 & w1).count_ones())
                        + ((m2 & w2).count_ones() + (m3 & w3).count_ones());
                    let a_ij = (inter as f32) * (inv_norm_i * inv_norm_cache[j]);
                    deg_slice[j] += a_ij;
                    acc_i0 += a_ij;
                    j += 1;
                }
                for j_tail in cached_upper.max(i + 1)..n {
                    let a_ij = mask_i.stigmergic_cosine_similarity_with_inv_norm(
                        inv_norm_i,
                        &block_masks[j_tail],
                    );
                    acc_i0 += a_ij;
                    deg_slice[j_tail] += a_ij;
                }
                deg_slice[i] = acc_i0 + acc_i1;
            }
            max_deg = branchless_max_f32(max_deg, deg_slice[i]);
        }
        Ok(branchless_flush_subnormal_f32(2.0 * max_deg))
    }

    /// Computes 3rd-order Chebyshev expansion coefficients $[c_0, c_1, c_2, c_3]$ in $\mathcal{O}(1)$
    /// using Gauss-Chebyshev nodes and division-free (`0 fdiv`) 4-way Montgomery inversion (Section 1.2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `epsilon <= 0.0`, or either is non-finite.
    #[inline]
    fn compute_tikhonov_chebyshev_coeffs_deg3(
        &self,
        lambda_max: f32,
        epsilon: f32,
    ) -> GcsoResult<[f32; CHEBYSHEV_COEFF_LEN]> {
        if !branchless_is_finite_f32(lambda_max)
            || !branchless_is_finite_f32(epsilon)
            || lambda_max <= 0.0
            || epsilon <= 0.0
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        const T1: [f32; 4] = [0.923_879_5, 0.382_683_43, -0.382_683_43, -0.923_879_5];
        const NODES_P1: [f32; 4] = [1.923_879_5, 1.382_683_4, 0.617_316_6, 0.076_120_496];
        const T2: [f32; 4] = [0.707_106_65, -0.707_106_77, -0.707_106_77, 0.707_106_65];
        const T3: [f32; 4] = [0.382_683_22, -0.923_879_5, 0.923_879_5, -0.382_683_22];

        let half_lmax = 0.5 * lambda_max;
        let d0 = half_lmax * NODES_P1[0] + epsilon;
        let d1 = half_lmax * NODES_P1[1] + epsilon;
        let d2 = half_lmax * NODES_P1[2] + epsilon;
        let d3 = half_lmax * NODES_P1[3] + epsilon;

        let d01 = d0 * d1;
        let d23 = d2 * d3;
        let prod_all = d01 * d23;
        let (g0, g1, g2, g3) = if branchless_is_finite_f32(prod_all) && prod_all > 1.0e-30 {
            let inv_all = branchless_inv_pos_f32(prod_all);
            let inv01 = inv_all * d23;
            let inv23 = inv_all * d01;
            (inv01 * d1, inv01 * d0, inv23 * d3, inv23 * d2)
        } else {
            (
                branchless_inv_pos_f32(d0),
                branchless_inv_pos_f32(d1),
                branchless_inv_pos_f32(d2),
                branchless_inv_pos_f32(d3),
            )
        };

        let c0 = (g0 + g1) + (g2 + g3);
        let c1 = (g0 * T1[0] + g1 * T1[1]) + (g2 * T1[2] + g3 * T1[3]);
        let c2 = (g0 * T2[0] + g1 * T2[1]) + (g2 * T2[2] + g3 * T2[3]);
        let c3 = (g0 * T3[0] + g1 * T3[1]) + (g2 * T3[2] + g3 * T3[3]);

        Ok([0.25 * c0, 0.5 * c1, 0.5 * c2, 0.5 * c3])
    }

    /// Evaluates a 3rd-order Chebyshev Polynomial Spectral Filter $\sum_{k=0}^3 c_k T_k(\tilde{\lambda})$
    /// using `0-fdiv` affine scaling and Horner-nested evaluation.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `lambda_max <= 0.0`, `lambda_max` is non-finite,
    /// `eigenvalues_in` is empty, or `eigenvalues_in.len() != filtered_out.len()`.
    #[inline]
    fn eval_chebyshev_spectral_filter_deg3(
        &self,
        eigenvalues_in: &AlignedSlice32<'_, f32>,
        filtered_out: &mut AlignedSliceMut32<'_, f32>,
        lambda_max: f32,
        cheb_coeffs: &[f32; CHEBYSHEV_COEFF_LEN],
    ) -> GcsoResult<()> {
        if !branchless_is_finite_f32(lambda_max)
            || lambda_max <= 0.0
            || eigenvalues_in.is_empty()
            || eigenvalues_in.len() != filtered_out.len()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let inv_half_lmax = 2.0 * branchless_inv_pos_f32(lambda_max);
        let k0 = cheb_coeffs[0] - cheb_coeffs[2];
        let k1 = cheb_coeffs[1] - 3.0 * cheb_coeffs[3];
        let k2 = 2.0 * cheb_coeffs[2];
        let k3 = 4.0 * cheb_coeffs[3];

        chebyshev_deg3_filter_slice_simd8(
            eigenvalues_in.as_slice(),
            filtered_out.as_mut_slice(),
            inv_half_lmax,
            k0,
            k1,
            k2,
            k3,
        );
        Ok(())
    }

    /// Evaluates the Hierarchical Tree Pseudo-ultrametric distance $d_{\mathcal{V}}(x, y) = \tanh(w_{\mathrm{key}} (l_{\max} - l_{\mathrm{lca}}))$ (Lemma 1.1).
    #[inline(always)]
    #[must_use]
    fn eval_pseudo_ultrametric_distance(&self, l_lca: u32, l_max: u32, w_key: f32) -> f32 {
        let diff = l_max.saturating_sub(l_lca) as f32;
        let safe_w =
            branchless_select_f32(branchless_is_finite_f32(w_key) & (w_key > 0.0), w_key, 0.0);
        branchless_tanh_f32(safe_w * diff)
    }

    /// Evaluates the cohomological obstruction residual $\mathbf{r}_{\mathrm{obs}} = \mathbf{r} - \delta_0 \boldsymbol{\phi} \in \mathrm{im}(\delta_0)^\perp$ (Section 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `cochain_residual_r` is empty or slice lengths differ.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if the squared residual norm overflows or is non-finite.
    #[inline]
    fn evaluate_cohomological_obstruction(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, bool)> {
        let n = cochain_residual_r.len();
        if n == 0 || n != exact_projection_delta0_phi.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let norm_sq = diff_norm_sq_f32_unrolled4(
            cochain_residual_r.as_slice(),
            exact_projection_delta0_phi.as_slice(),
        );
        if !branchless_is_finite_f32(norm_sq) {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }
        let mean_obs = branchless_flush_subnormal_f32(norm_sq * fast_inv_len_f32(n));
        Ok((mean_obs, mean_obs > obstruction_threshold))
    }

    /// Evaluates the Hodge obstruction residual $\mathbf{r}_{\mathrm{obs}} \in H^1(K; \mathcal{F})$
    /// and normalized dual Cellular Sheaf Homology pairing $\frac{\vert\langle \mathbf{r}_{\mathrm{obs}}, \boldsymbol{\gamma} \rangle\vert}{\|\mathbf{r}_{\mathrm{obs}}\|_2 \|\boldsymbol{\gamma}\|_2} \in [0, 1]$ (Section 1.1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if any slice is empty or lengths differ.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if any accumulated norm or pairing inner product is non-finite.
    #[inline]
    fn evaluate_hodge_obstruction_and_dual_pairing(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        dual_homology_cycle_gamma: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, f32, bool)> {
        let n = cochain_residual_r.len();
        if n == 0 || exact_projection_delta0_phi.len() != n || dual_homology_cycle_gamma.len() != n
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let (obs_sq, pair_dot, gam_sq) = hodge_obstruction_and_dual_pairing_f32(
            cochain_residual_r.as_slice(),
            exact_projection_delta0_phi.as_slice(),
            dual_homology_cycle_gamma.as_slice(),
        );

        if !branchless_is_finite_f32(obs_sq)
            || !branchless_is_finite_f32(pair_dot)
            || !branchless_is_finite_f32(gam_sq)
        {
            return Err(GCSO_ERROR_EDBC_SINGULARITY);
        }

        let mean_obs = branchless_flush_subnormal_f32(obs_sq * fast_inv_len_f32(n));
        let prod_sq = obs_sq * gam_sq;
        let valid_pair = prod_sq > 1.0e-24;
        let inv_denom = branchless_rsqrt_f32(branchless_select_f32(valid_pair, prod_sq, 1.0));
        let norm_pairing = branchless_select_f32(
            valid_pair,
            branchless_clamp_f32(branchless_abs_f32(pair_dot) * inv_denom, 0.0, 1.0),
            0.0,
        );

        Ok((mean_obs, norm_pairing, mean_obs > obstruction_threshold))
    }

    /// Detects cohomological obstruction $H^1(K; \mathcal{F})$ and records the offending phase
    /// pattern into `bloom_filter` while returning the phase-conjugate anti-phase vector ($-\boldsymbol{\Delta\theta}$).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths mismatch or are empty.
    /// Returns [`GCSO_ERROR_EDBC_SINGULARITY`] if residual norms are non-finite.
    #[inline]
    fn detect_and_record_obstruction_repulsion(
        &self,
        cochain_residual_r: &AlignedSlice32<'_, f32>,
        exact_projection_delta0_phi: &AlignedSlice32<'_, f32>,
        obstruction_threshold: f32,
        active_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(f32, bool, [gcso_q7_t; MAX_ATTENTION_HEADS])> {
        let (mean_obs, is_obstructed) = self.evaluate_cohomological_obstruction(
            cochain_residual_r,
            exact_projection_delta0_phi,
            obstruction_threshold,
        )?;
        if is_obstructed {
            bloom_filter.insert_repulsive_pattern(active_phases);
            Ok((mean_obs, true, branchless_q7_conjugate_64(active_phases)))
        } else {
            Ok((mean_obs, false, *active_phases))
        }
    }

    /// Evaluates the **Joint Probability Chain Rule Log-Likelihood**
    /// $\ln P(M_1, \dots, M_T) = \sum_{t=1}^{T-1} \ln P(M_t \mid M_{t-1})$ and **$S^1$ Sheaf
    /// Holonomy Obstruction Energy** ($H^1(K; \mathcal{F})$) across `trail_chain` in $\mathcal{O}(T)$ time.
    ///
    /// Returns `(joint_log_prob_sum, holonomy_obstruction_energy, is_obstructed)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or `obstruction_threshold` is non-finite.
    #[inline]
    fn evaluate_trail_chain_cohomology_and_joint_log_prob(
        &self,
        trail_chain: &[gcso_pointer_trail_t],
        obstruction_threshold: f32,
    ) -> GcsoResult<(f32, f32, bool)> {
        let len = trail_chain.len();
        if len == 0 || !branchless_is_finite_f32(obstruction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if len == 1 {
            return Ok((0.0, 0.0, false));
        }

        let mut joint_log_prob = 0.0_f32;
        let mut coboundary_defect_sum = 0.0_f32;
        let first_w = *trail_phase_words_ref(&trail_chain[0]);
        let mut prev_w = first_w;
        for trail in &trail_chain[1..len] {
            let cur_w = *trail_phase_words_ref(trail);
            let (coh, log_p) = branchless_q7_transition_log_prob_words_64(&prev_w, &cur_w);
            joint_log_prob += log_p;
            coboundary_defect_sum += branchless_relu_f32(0.5 * (1.0 - coh));
            prev_w = cur_w;
        }

        let mean_step_defect = coboundary_defect_sum * fast_inv_len_f32(len - 1);
        let loop_coh = coherence_from_word_arrays_64(&first_w, &prev_w);
        let loop_defect = branchless_relu_f32(0.5 * (1.0 - loop_coh));
        let holonomy_energy =
            branchless_flush_subnormal_f32(0.5 * (mean_step_defect + loop_defect));
        let is_obstructed = holonomy_energy > branchless_relu_f32(obstruction_threshold);

        Ok((
            branchless_flush_subnormal_f32(joint_log_prob),
            holonomy_energy,
            is_obstructed,
        ))
    }

    /// Bridges all consecutive inter-memory gaps (**Limina**) across `trail_chain` via
    /// [`StigmergicTrailOps::bridge_liminal_hallucinated_trail`], recording any self-contradictory
    /// transitions ($H^1(K; \mathcal{F}) \neq 0$) into `bloom_filter`.
    ///
    /// Returns `(total_joint_log_prob, repulsed_transition_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `trail_chain` is empty or `contradiction_threshold` is non-finite.
    #[inline]
    fn bridge_and_filter_liminal_trail_chain(
        &self,
        trail_chain: &mut [gcso_pointer_trail_t],
        interpolation_u7: u8,
        contradiction_threshold: f32,
        bloom_filter: &mut ContrastivePhaseBloomFilter,
    ) -> GcsoResult<(f32, u32)> {
        let len = trail_chain.len();
        if len == 0 || !branchless_is_finite_f32(contradiction_threshold) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut total_log_prob = 0.0_f32;
        let mut repulsed_count = 0u32;

        for idx in 1..len {
            let (head, tail) = trail_chain.split_at_mut(idx);
            let prev_ref = &head[idx - 1];
            let cur_mut = &mut tail[0];
            let (_coh, log_p, repulsed) = cur_mut.bridge_liminal_hallucinated_trail(
                prev_ref,
                interpolation_u7,
                contradiction_threshold,
            );
            total_log_prob += log_p;
            if repulsed {
                bloom_filter.insert_repulsive_pattern_words_weighted(
                    trail_phase_words_ref(cur_mut),
                    1,
                );
                repulsed_count = repulsed_count.wrapping_add(1);
            }
        }
        Ok((branchless_flush_subnormal_f32(total_log_prob), repulsed_count))
    }
}

// ===================================================================
// 2. Hierarchical Tree Table (QDPT / CPT) & Pointer Mode Cascading
// ===================================================================

/// Trait for Quantized Depth Pointer Table (QDPT) and Cascade Pointer Table (CPT) (**Micro / Mezzo Level**).
pub trait HierarchicalTreeTable: BranchlessKernel + Send + Sync {
    /// Quantizes continuous pseudo-ultrametric depth $d_{\mathcal{V}} \in [0, 1)$ into discrete QDPT level
    /// $\bar{D}_{\mathcal{V},\mathrm{quantized}}$ with hysteresis band $\delta_{\mathrm{hyst}}$ (Section 1.3).
    #[inline(always)]
    #[must_use]
    fn quantize_depth_hysteresis(
        &self,
        d_v_distance: f32,
        current_level: u32,
        max_level: u32,
        hysteresis_band: f32,
    ) -> u32 {
        let scaled = branchless_clamp_f32(d_v_distance, 0.0, 1.0) * (max_level as f32);
        branchless_hysteresis_quantize_depth(scaled, current_level, hysteresis_band, max_level)
    }

    /// Executes an $\mathcal{O}(1)$ cascade pointer lookup combining parent and child phase deltas.
    #[inline(always)]
    #[must_use]
    fn cascade_tagged_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        child_addr_offset: u64,
        child_phase_delta: gcso_q7_t,
    ) -> gcso_tagged_ptr_t {
        let new_addr =
            parent_ptr.unpack_addr().wrapping_add(child_addr_offset) & gcso_tagged_ptr_t::ADDR_MASK;
        let combined_phase = branchless_q7_add_sat(parent_ptr.unpack_phase(), child_phase_delta);
        gcso_tagged_ptr_t::pack(new_addr, parent_ptr.unpack_tag(), combined_phase)
    }

    /// Executes an $\mathcal{O}(1)$ cascade pointer transition merging `child_action_mode_or_mask`
    /// and binding to `child_attractor_slot` (`0..=7`).
    #[inline(always)]
    #[must_use]
    fn cascade_action_slot_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        child_addr_offset: u64,
        child_attractor_slot: u8,
        child_action_mode_or_mask: u8,
        child_phase_delta: gcso_q7_t,
    ) -> gcso_tagged_ptr_t {
        let new_addr =
            parent_ptr.unpack_addr().wrapping_add(child_addr_offset) & gcso_tagged_ptr_t::ADDR_MASK;
        let merged_mode = parent_ptr.action_mode_flags() | child_action_mode_or_mask;
        let tag = pack_action_slot_tag(child_attractor_slot, merged_mode);
        let combined_phase = branchless_q7_add_sat(parent_ptr.unpack_phase(), child_phase_delta);
        gcso_tagged_ptr_t::pack(new_addr, tag, combined_phase)
    }

    /// Evaluates 256-bit dyadic distance $d_{\mathcal{V}}(x, y) \in [0, 1)$, quantizes it into a
    /// discrete QDPT level, and cascades `parent_ptr` in $\mathcal{O}(1)$.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `qdpt_addr_offsets` is empty or its length does
    /// not match `qdpt_phase_deltas.len()`.
    #[inline(always)]
    fn resolve_dyadic_qdpt_and_cascade_ptr(
        &self,
        parent_ptr: gcso_tagged_ptr_t,
        current_mask: &gcso_paged_bitmask_t,
        anchor_mask: &gcso_paged_bitmask_t,
        w_key: f32,
        current_level: u32,
        hysteresis_band: f32,
        qdpt_addr_offsets: &[u64],
        qdpt_phase_deltas: &[gcso_q7_t],
    ) -> GcsoResult<(u32, f32, gcso_tagged_ptr_t)> {
        let table_len = qdpt_addr_offsets.len();
        if table_len == 0 || table_len != qdpt_phase_deltas.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let max_level = (table_len - 1) as u32;
        let d_v = eval_bitmask_dyadic_ultrametric(current_mask, anchor_mask, w_key);
        let clamped_cur =
            branchless_select_u32(current_level <= max_level, current_level, max_level);
        let level = self.quantize_depth_hysteresis(d_v, clamped_cur, max_level, hysteresis_band);
        let idx = (level as usize).min(table_len - 1);
        let cascaded =
            self.cascade_tagged_ptr(parent_ptr, qdpt_addr_offsets[idx], qdpt_phase_deltas[idx]);
        Ok((level, d_v, cascaded))
    }
}

