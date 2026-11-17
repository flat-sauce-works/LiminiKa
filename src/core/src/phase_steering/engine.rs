// name: src/core/src/phase_steering/engine.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compact 16B [Pod] [PhaseSteeringEngine] and core RoPE trait delegations.

#[allow(unused_imports)]
use super::closed_loop::*;
use super::*;

// ===================================================================
// 2. Primary DPSR Phase-Steering & Sub-Head Routing Engine (`16B` Pod)
// ===================================================================

/// Primary Dynamic Phase-Shifted RoPE (DPSR), RIPA, and QDPS execution engine (`16B` aligned, [`Pod`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSteeringEngine {
    /// Dimension per attention head ($d_{\mathrm{head}}$, positive even integer).
    pub(super) head_dim: u32,
    /// Total number of attention heads ($H > 0$).
    pub(super) num_heads: u32,
    /// Default RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    pub(super) ripa_clamp_max_rad: f32,
    /// Default QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub(super) qdps_min_step_rad: f32,
}

// SAFETY: `PhaseSteeringEngine` is `#[repr(C, align(16))]` containing four 32-bit primitive fields totaling 16B.
unsafe impl Pod for PhaseSteeringEngine {}

const _: () = {
    assert!(size_of::<PhaseSteeringEngine>() == 16);
    assert!(align_of::<PhaseSteeringEngine>() == 16);
    assert!(offset_of!(PhaseSteeringEngine, head_dim) == 0);
    assert!(offset_of!(PhaseSteeringEngine, num_heads) == 4);
    assert!(offset_of!(PhaseSteeringEngine, ripa_clamp_max_rad) == 8);
    assert!(offset_of!(PhaseSteeringEngine, qdps_min_step_rad) == 12);
    assert!(NUM_SWARM_MICRO_CELLS * HEADS_PER_SWARM_CELL == MAX_ATTENTION_HEADS);
};

impl Default for PhaseSteeringEngine {
    #[inline(always)]
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_DPSR_HEAD_DIM,
            num_heads: DEFAULT_DPSR_NUM_HEADS,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
        }
    }
}

impl PhaseSteeringEngine {
    /// Creates a new [`PhaseSteeringEngine`] instance with validated `(head_dim, num_heads)` configuration.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// or `head_dim * num_heads` overflows `usize`.
    #[inline(always)]
    pub const fn new(head_dim: u32, num_heads: u32) -> GcsoResult<Self> {
        let valid = (head_dim != 0) & ((head_dim & 1) == 0) & (num_heads != 0);
        if !valid {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if (head_dim as usize)
            .checked_mul(num_heads as usize)
            .is_none()
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(Self {
            head_dim,
            num_heads,
            ripa_clamp_max_rad: DEFAULT_RIPA_CLAMP_MAX_RAD,
            qdps_min_step_rad: DEFAULT_QDPS_MIN_STEP_RAD,
        })
    }

    /// Constructs a [`PhaseSteeringEngine`] from a C-ABI [`gcso_config_t`] descriptor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `head_dim` is zero or odd, `num_heads` is zero,
    /// `ripa_clamp_max_rad <= 0.0`, `qdps_min_step_rad < 0.0`, or thresholds are non-finite.
    #[inline(always)]
    pub const fn from_config(config: &gcso_config_t) -> GcsoResult<Self> {
        let mut engine = match Self::new(config.head_dim, config.num_heads) {
            Ok(e) => e,
            Err(err) => return Err(err),
        };
        let valid_ripa =
            branchless_is_finite_f32(config.ripa_clamp_max_rad) & (config.ripa_clamp_max_rad > 0.0);
        let valid_qdps =
            branchless_is_finite_f32(config.qdps_min_step_rad) & (config.qdps_min_step_rad >= 0.0);
        if !(valid_ripa & valid_qdps) {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        engine.ripa_clamp_max_rad = config.ripa_clamp_max_rad;
        engine.qdps_min_step_rad = config.qdps_min_step_rad;
        Ok(engine)
    }

    /// Returns the configured RIPA soft-bounded phase clamp limit in radians ($\theta_{\max}$).
    #[inline(always)]
    pub const fn ripa_clamp_max_rad(&self) -> f32 {
        self.ripa_clamp_max_rad
    }

    /// Returns the configured QDPS minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    pub const fn qdps_min_step_rad(&self) -> f32 {
        self.qdps_min_step_rad
    }

    /// Returns a copy of `self` with an updated RIPA soft-clamping limit (`ripa_clamp_max_rad`).
    #[inline(always)]
    pub const fn with_ripa_clamp_max_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.ripa_clamp_max_rad = branchless_select_f32(valid, max_rad, self.ripa_clamp_max_rad);
        self
    }

    /// Returns a copy of `self` with an updated QDPS step cutoff threshold (`qdps_min_step_rad`).
    #[inline(always)]
    pub const fn with_qdps_min_step_rad(mut self, min_step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(min_step_rad) & (min_step_rad >= 0.0);
        self.qdps_min_step_rad = branchless_select_f32(valid, min_step_rad, self.qdps_min_step_rad);
        self
    }

    /// Returns the total number of `f32` elements expected per single-token Query/Key/Value tensor.
    #[inline(always)]
    pub const fn expected_tensor_len(&self) -> usize {
        (self.head_dim as usize) * (self.num_heads as usize)
    }

    /// Applies pointer-gated 8-Cell Micro-Swarm Quorum-Gated RIPA Phase Steering with configurable
    /// `unresolved_damping_u7`, delegating directly to
    /// [`PhaseSteering::apply_pointer_swarm_quorum_gated_steering_safe`].
    ///
    /// # Errors
    /// Propagates any error from [`PhaseSteering::apply_pointer_swarm_quorum_gated_steering_safe`].
    #[inline(always)]
    pub fn apply_swarm_quorum_gated_steering_safe(
        &self,
        tagged_ptr: gcso_tagged_ptr_t,
        swarm_quorum_mask: u8,
        unresolved_damping_u7: u8,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_pointer_swarm_quorum_gated_steering_safe(
            tagged_ptr,
            swarm_quorum_mask,
            unresolved_damping_u7,
            phase_deltas,
            query_tensor,
            key_tensor,
            max_rad,
        )
    }

    /// Steers `query_tensor` (and optional `key_tensor`) in registers using `self.qdps_min_step_rad`
    /// and `self.ripa_clamp_max_rad` configured on this [`PhaseSteeringEngine`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if tensor dimensions or non-aliasing invariants fail.
    #[inline(always)]
    pub fn steer_qk_with_defaults(
        &self,
        raw_phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<()> {
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            self.ripa_clamp_max_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;
        filter_qdps_and_steer_from_phases_64(
            raw_phases,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok(())
    }

    /// Folds an arbitrary slice of anchors (`anchor_phases`, `anchor_base_energies`) on the
    /// **Single Unified Attractor Manifold** into `trail` (updating `trail.target_anchor_id` and
    /// `trail.cluster_id`) and steers `query_tensor` (and optional `key_tensor`) using
    /// `self.qdps_min_step_rad` and `self.ripa_clamp_max_rad`.
    ///
    /// Returns `(final_pull_force, mean_coherence, active_anchor_count)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `anchor_phases.len() != anchor_base_energies.len()`
    /// or tensor dimensions fail.
    #[inline(always)]
    pub fn steer_unified_field_with_defaults(
        &self,
        trail: &mut gcso_pointer_trail_t,
        anchor_phases: &[[gcso_q7_t; MAX_ATTENTION_HEADS]],
        anchor_base_energies: &[f32],
        repulsion_phases_and_strength: Option<(&[gcso_q7_t; MAX_ATTENTION_HEADS], f32)>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(f32, f32, u32)> {
        if anchor_phases.len() != anchor_base_energies.len() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (head_dim, num_heads) = validate_closed_loop_tensors(
            self,
            self.ripa_clamp_max_rad,
            query_tensor,
            key_tensor.as_deref(),
        )?;
        let (pull, mean_coh, active_cnt, raw_w) = trail
            .apply_unified_field_anchor_slice_resonance_to_words(
                anchor_phases,
                anchor_base_energies,
                repulsion_phases_and_strength,
            );
        filter_qdps_and_steer_from_words_64(
            &raw_w,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            head_dim,
            num_heads,
            query_tensor,
            key_tensor,
        );
        Ok((pull, mean_coh, active_cnt))
    }

    /// Executes [`PhaseSteeringClosedLoop::select_rewire_and_steer_bayesian_pointer_trail_safe`]
    /// using `self.qdps_min_step_rad` and `self.ripa_clamp_max_rad` configured on this [`PhaseSteeringEngine`].
    ///
    /// Returns `(selected_trail_id, bayesian_score, transition_coherence, transition_log_prob, found_coherent_transition)`.
    ///
    /// # Errors
    /// Propagates any validation error from [`PhaseSteeringClosedLoop::select_rewire_and_steer_bayesian_pointer_trail_safe`].
    #[inline(always)]
    pub fn select_rewire_and_steer_bayesian_with_defaults(
        &self,
        src_trail: &mut gcso_pointer_trail_t,
        candidate_trails: &mut [gcso_pointer_trail_t],
        phase_interference_gain: f32,
        stigmergic_prior_gain: f32,
        pointer_link_bonus: f32,
        repulsion_penalty: f32,
        interpolation_u7: u8,
        contradiction_threshold: f32,
        attention_scores: Option<&mut AlignedSliceMut32<'_, f32>>,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(u32, f32, f32, f32, bool)> {
        self.select_rewire_and_steer_bayesian_pointer_trail_safe(
            src_trail,
            candidate_trails,
            phase_interference_gain,
            stigmergic_prior_gain,
            pointer_link_bonus,
            repulsion_penalty,
            interpolation_u7,
            contradiction_threshold,
            self.qdps_min_step_rad,
            self.ripa_clamp_max_rad,
            attention_scores,
            query_tensor,
            key_tensor,
        )
    }

    /// Executes [`PhaseSteeringClosedLoop::steer_and_kmeans_crystallize_gateway_signal_safe`] with
    /// default DP-Means split coherence ([`DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE`] = `0.35`)
    /// and unresolved cell damping (`64` = `0.5x`).
    ///
    /// Returns `(GatewayFrame, CausalProvenanceReport, assigned_cluster_idx, assigned_anchor_id, centroid_coherence, updated_gravity_mass, spawned_new_cluster)`.
    ///
    /// # Errors
    /// Propagates any error from [`PhaseSteeringClosedLoop::steer_and_kmeans_crystallize_gateway_signal_safe`].
    #[inline(always)]
    pub fn steer_and_kmeans_crystallize_with_defaults<S: ManifoldGatewaySignal + ?Sized>(
        &self,
        signal: &S,
        trail: &mut gcso_pointer_trail_t,
        centroids: &mut [CrystallizedAttractorRecord],
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        logits_opt: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<(
        GatewayFrame,
        CausalProvenanceReport,
        usize,
        u32,
        f32,
        f32,
        bool,
    )> {
        self.steer_and_kmeans_crystallize_gateway_signal_safe(
            signal,
            trail,
            centroids,
            DEFAULT_TOROIDAL_DP_MEANS_SPLIT_COHERENCE,
            64,
            query_tensor,
            key_tensor,
            logits_opt,
        )
    }
}

// ===================================================================
// 3. Minimal Trait Delegations (Inheriting All `rope.rs` Default Implementations)
// ===================================================================

impl PhaseSteering for PhaseSteeringEngine {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.head_dim
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.num_heads
    }
}

impl EntropyGatedPhaseLookup for PhaseSteeringEngine {}

impl DynamicLayerSkipper for PhaseSteeringEngine {}

impl SubHeadRouter for PhaseSteeringEngine {}

impl VisualAttentionSteering for PhaseSteeringEngine {}
