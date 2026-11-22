// name: src/core/src/stigmergy/gate.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Layer 2 (`Stigmergy`) scoped execution gates ([`PhaseSteeringEngine`] and
//! [`ResidualAdapterEngine`]) bridging `liminal::axiom` (`Indivisible<Active, _>`) with pure
//! `foundation` RoPE and Rank-1 residual kernels.

use core::mem::{align_of, offset_of, size_of};

use crate::foundation::aligned::{AlignedSlice16, AlignedSlice32, AlignedSliceMut32};
use crate::foundation::phase_ops::circular_r_bar_from_words_64;
use crate::foundation::pod::{gcso_rank1_residual_descriptor_t, Pod};
use crate::foundation::projection::project_bytes_swar_to_words_64;
use crate::foundation::rank1_kernels::eval_rank1_slices_fused;
use crate::foundation::rope::{
    PhaseSteering, PhaseSteeringGeometryExt, RopeSteeringTarget, DEFAULT_PHASE_STEER_HEAD_DIM,
    DEFAULT_PHASE_STEER_NUM_HEADS,
};
use crate::foundation::scalar_math::{
    branchless_clamp_f32, branchless_is_finite_f32, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_STEP_MIN_RAD,
};
use crate::liminal::{
    branchless_select_f32, gcso_config_t, gcso_q7_t, AbstractDomain, Active, GcsoResult,
    Indivisible, GCSO_ERROR_INVALID_ARGUMENT, HEADS_PER_STIGMERGIC_GRANULE, MAX_ATTENTION_HEADS,
    NUM_STIGMERGIC_GRANULES,
};

/// Zero-byte domain-tagged scalar scale factor for Rank-1 residual injection (`Active` phase).
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualScale(pub f32);

impl AbstractDomain for ResidualScale {
    const DOMAIN_ID: &'static str = "stigmergy.residual_scale";
}

/// Zero-byte domain-tagged 16-byte-aligned Q7 phase-delta slice for RoPE steering (`Active` phase).
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct PhaseDeltaSlice<'a>(pub AlignedSlice16<'a, gcso_q7_t>);

impl AbstractDomain for PhaseDeltaSlice<'_> {
    const DOMAIN_ID: &'static str = "stigmergy.phase_delta_slice";
}

const _: () = {
    assert!(size_of::<Indivisible<Active, ResidualScale>>() == size_of::<f32>());
    assert!(align_of::<Indivisible<Active, ResidualScale>>() == align_of::<f32>());
    assert!(
        size_of::<Indivisible<Active, PhaseDeltaSlice<'static>>>()
            == size_of::<AlignedSlice16<'static, gcso_q7_t>>()
    );
    assert!(
        align_of::<Indivisible<Active, PhaseDeltaSlice<'static>>>()
            == align_of::<AlignedSlice16<'static, gcso_q7_t>>()
    );
};

// ===================================================================
// 1. Primary Phase-Steering Engine (`16B` Pod)
// ===================================================================

/// Primary dynamic phase-shifted RoPE execution engine (`PhaseSteering`, `PhaseClamp`, `PhaseStepFilter`) (`16B` aligned, [`Pod`]).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSteeringEngine {
    /// Dimension per attention head ($d_{\mathrm{head}}$, positive even integer).
    pub(crate) head_dim: u32,
    /// Total number of attention heads ($H > 0$).
    pub(crate) num_heads: u32,
    /// Default soft-bounded (`PhaseClamp`) limit in radians ($\theta_{\max}$).
    pub(crate) phase_clamp_max_rad: f32,
    /// Default `PhaseStepFilter` minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    pub(crate) phase_step_min_rad: f32,
}

// SAFETY: `PhaseSteeringEngine` is `#[repr(C, align(16))]` containing four 32-bit primitive fields totaling 16B.
unsafe impl Pod for PhaseSteeringEngine {}

const _: () = {
    assert!(size_of::<PhaseSteeringEngine>() == 16);
    assert!(align_of::<PhaseSteeringEngine>() == 16);
    assert!(offset_of!(PhaseSteeringEngine, head_dim) == 0);
    assert!(offset_of!(PhaseSteeringEngine, num_heads) == 4);
    assert!(offset_of!(PhaseSteeringEngine, phase_clamp_max_rad) == 8);
    assert!(offset_of!(PhaseSteeringEngine, phase_step_min_rad) == 12);
    assert!(NUM_STIGMERGIC_GRANULES * HEADS_PER_STIGMERGIC_GRANULE == MAX_ATTENTION_HEADS);
};

impl Default for PhaseSteeringEngine {
    #[inline(always)]
    fn default() -> Self {
        Self {
            head_dim: DEFAULT_PHASE_STEER_HEAD_DIM,
            num_heads: DEFAULT_PHASE_STEER_NUM_HEADS,
            phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
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
            phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
            phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
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
        engine.phase_clamp_max_rad = config.ripa_clamp_max_rad;
        engine.phase_step_min_rad = config.qdps_min_step_rad;
        Ok(engine)
    }

    /// Returns the configured soft-bounded (`PhaseClamp`) limit in radians ($\theta_{\max}$).
    #[inline(always)]
    pub const fn phase_clamp_max_rad(&self) -> f32 {
        self.phase_clamp_max_rad
    }

    /// Returns the configured `PhaseStepFilter` minimum phase step cutoff in radians ($\Delta\theta_{\mathrm{min\_step}}$).
    #[inline(always)]
    pub const fn phase_step_min_rad(&self) -> f32 {
        self.phase_step_min_rad
    }

    /// Returns a copy of `self` with an updated `PhaseClamp` soft-clamping limit (`ripa_clamp_max_rad`).
    #[inline(always)]
    pub const fn with_phase_clamp_max_rad(mut self, max_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(max_rad) & (max_rad > 0.0);
        self.phase_clamp_max_rad = branchless_select_f32(valid, max_rad, self.phase_clamp_max_rad);
        self
    }

    /// Returns a copy of `self` with an updated `PhaseStepFilter` step cutoff threshold (`qdps_min_step_rad`).
    #[inline(always)]
    pub const fn with_phase_step_min_rad(mut self, min_step_rad: f32) -> Self {
        let valid = branchless_is_finite_f32(min_step_rad) & (min_step_rad >= 0.0);
        self.phase_step_min_rad =
            branchless_select_f32(valid, min_step_rad, self.phase_step_min_rad);
        self
    }

    /// Returns the total number of `f32` elements expected per single-token Query/Key/Value tensor.
    #[inline(always)]
    pub const fn expected_tensor_len(&self) -> usize {
        (self.head_dim as usize) * (self.num_heads as usize)
    }

    /// Non-generic [`Active`] closure entry point for [`PhaseSteering::apply_phase_steering`].
    ///
    /// Keeps the underlying AVX2/scalar RoPE kernel strictly non-generic (`0 B` L1I duplication)
    /// while statically excluding [`crate::liminal::Unmarked`], [`crate::liminal::Bottom`], and
    /// unclassified origins at compile time.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if slice lengths are insufficient.
    #[inline(always)]
    pub fn apply_phase_steering_active(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Indivisible<Active, PhaseDeltaSlice<'_>>,
    ) -> GcsoResult<()> {
        self.apply_phase_steering(tensor, &phase_deltas.as_ref().0)
    }

    /// Non-generic [`Indivisible<Active, _>`] closure entry point for
    /// [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if geometry or slice lengths are invalid.
    #[inline(always)]
    pub fn apply_phase_steering_for_geometry_safe_active<G: RopeSteeringTarget + ?Sized>(
        &self,
        geometry: &G,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Indivisible<Active, PhaseDeltaSlice<'_>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_for_geometry_safe(
            geometry,
            tensor,
            &phase_deltas.as_ref().0,
            max_rad,
        )
    }
}

/// Blanket extension trait adding [`Indivisible<Active, PhaseDeltaSlice<'_>>`] gated RoPE
/// steering entry points to any [`PhaseSteeringGeometryExt`] implementor without altering pure
/// Level 1 traits.
pub trait PhaseSteeringActiveExt: PhaseSteeringGeometryExt {
    /// Applies [`PhaseSteering::apply_phase_steering_safe`] only after `phase_deltas` carries an
    /// [`Indivisible<Active, _>`] sealed closure certificate.
    ///
    /// # Errors
    /// Propagates any validation or alignment error from
    /// [`PhaseSteering::apply_phase_steering_safe`].
    #[inline(always)]
    fn apply_phase_steering_safe_active(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Indivisible<Active, PhaseDeltaSlice<'_>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_safe(query_tensor, &phase_deltas.as_ref().0, max_rad)
    }

    /// Applies [`PhaseSteering::apply_phase_steering_fused_safe`] only after `phase_deltas`
    /// carries an [`Indivisible<Active, _>`] sealed closure certificate.
    ///
    /// # Errors
    /// Propagates any validation, aliasing, or alignment error from
    /// [`PhaseSteering::apply_phase_steering_fused_safe`].
    #[inline(always)]
    fn apply_phase_steering_fused_safe_active(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Indivisible<Active, PhaseDeltaSlice<'_>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_fused_safe(
            query_tensor,
            key_tensor,
            &phase_deltas.as_ref().0,
            max_rad,
        )
    }

    /// Applies [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`] only after
    /// `phase_deltas` carries an [`Indivisible<Active, _>`] sealed closure certificate.
    ///
    /// # Errors
    /// Propagates any validation, geometry, or alignment error from
    /// [`PhaseSteeringGeometryExt::apply_phase_steering_for_geometry_safe`].
    #[inline(always)]
    fn apply_phase_steering_for_geometry_safe_active<G: RopeSteeringTarget + ?Sized>(
        &self,
        geometry: &G,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: Indivisible<Active, PhaseDeltaSlice<'_>>,
        max_rad: f32,
    ) -> GcsoResult<()> {
        self.apply_phase_steering_for_geometry_safe(
            geometry,
            query_tensor,
            &phase_deltas.as_ref().0,
            max_rad,
        )
    }
}

impl<T: PhaseSteeringGeometryExt + ?Sized> PhaseSteeringActiveExt for T {}

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

// ===================================================================
// 2. Rank-1 Residual Adapter Engine (`64B` Pod)
// ===================================================================

/// 64-byte cacheline-aligned Plain Old Data (`Pod`) descriptor and execution engine for
/// Rank-1 outer-product residual adapters.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualAdapterEngine {
    /// Embedded 64-byte C-ABI Rank-1 residual descriptor (`offset 0..64`).
    pub descriptor: gcso_rank1_residual_descriptor_t,
}

// SAFETY: `ResidualAdapterEngine` is a `#[repr(C, align(64))]` wrapper around the 64-byte `Pod` `gcso_rank1_residual_descriptor_t`.
unsafe impl Pod for ResidualAdapterEngine {}

impl Default for ResidualAdapterEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(1.0, 0, 1)
    }
}

impl ResidualAdapterEngine {
    /// Constructs a new 64-byte aligned [`ResidualAdapterEngine`] with canonical C-ABI descriptor fields.
    #[inline(always)]
    #[must_use]
    pub const fn new(scale_factor: f32, layer_idx: u32, rank: u32) -> Self {
        Self {
            descriptor: gcso_rank1_residual_descriptor_t {
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

    /// Validates the embedded `gcso_rank1_residual_descriptor_t` fields (`rank > 0` and finite positive `scale_factor`).
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

    /// Validated Rank-1 outer-product residual injection ($\mathbf{y} \gets \mathbf{y} + s \cdot \mathrm{diag}(\mathbf{g})\mathbf{u}(\mathbf{v}^T \mathbf{x})$)
    /// gated by an [`Indivisible<Active, _>`] sealed closure certificate.
    ///
    /// Statically excludes [`crate::liminal::Unmarked`], [`crate::liminal::Bottom`], and
    /// unclassified origins from injecting Rank-1 hidden-state residuals while keeping the
    /// underlying AVX2/scalar kernel non-generic (`0 B` L1I duplication).
    ///
    /// # Errors
    /// Propagates any validation or numerical singularity error from [`eval_rank1_slices_fused`].
    #[inline(always)]
    pub fn eval_rank1_slices_fused_active(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        u_vec: &AlignedSlice32<'_, f32>,
        v_vec: &AlignedSlice32<'_, f32>,
        gain_vec: Option<&AlignedSlice32<'_, f32>>,
        scale_factor: Indivisible<Active, ResidualScale>,
    ) -> GcsoResult<()> {
        eval_rank1_slices_fused(
            y_inout,
            x_in,
            u_vec,
            v_vec,
            gain_vec,
            scale_factor.as_ref().0,
        )
    }

    /// Applies a binary persona phase modulation patch (`patch_data`) onto this engine's scale factor.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `patch_data` is empty.
    #[inline]
    pub fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
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
    assert!(size_of::<ResidualAdapterEngine>() == 64);
    assert!(align_of::<ResidualAdapterEngine>() == 64);
    assert!(offset_of!(ResidualAdapterEngine, descriptor) == 0);
};

#[cfg(test)]
mod tests {
    use core::f32::consts::PI;

    use super::*;
    use crate::foundation::aligned::{AlignedBuffer128, AlignedBuffer16, AlignedBuffer32};
    use crate::foundation::scalar_math::{branchless_sincos_q7, branchless_sqrt_f32};
    use crate::liminal::{Scoped, ScopedCut};

    #[repr(align(32))]
    struct Align32<T>(T);

    #[test]
    fn test_active_closure_gate_matches_non_generic_rope_kernel_bit_exact() {
        let engine = PhaseSteeringEngine::new(16, 4).unwrap();
        let mut raw_buf = Align32([0.0_f32; 64]);
        let mut gated_buf = Align32([0.0_f32; 64]);
        for (i, slot) in raw_buf.0.iter_mut().enumerate() {
            *slot = ((i as f32) * 0.13) - 2.0;
        }
        gated_buf.0 = raw_buf.0;

        let deltas_raw = Align32([12_i8, -18, 27, -9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let deltas_slice = AlignedSlice16::new(&deltas_raw.0[..4]).unwrap();

        {
            let mut raw_mut = AlignedSliceMut32::new(&mut raw_buf.0).unwrap();
            engine
                .apply_phase_steering(&mut raw_mut, &deltas_slice)
                .unwrap();
        }
        {
            let mut gated_mut = AlignedSliceMut32::new(&mut gated_buf.0).unwrap();
            let active = match Scoped::from_boundary(PhaseDeltaSlice(deltas_slice))
                .cut_by_lacuna_q7(true, 12, 4)
            {
                ScopedCut::Exterior(ext) => ext.into_active(),
                _ => panic!("expected exterior active cut"),
            };
            assert_eq!(active.domain_id(), "stigmergy.phase_delta_slice");
            engine
                .apply_phase_steering_active(&mut gated_mut, active)
                .unwrap();
        }

        for (lhs, rhs) in raw_buf.0.iter().zip(gated_buf.0.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }

    #[test]
    fn test_phase_steering_engine_creation_aligned_dispatch_and_pod_layout() {
        assert!(PhaseSteeringEngine::new(128, 32).is_ok());
        assert!(PhaseSteeringEngine::new(127, 32).is_err());
        assert!(PhaseSteeringEngine::new(0, 32).is_err());
        assert!(PhaseSteeringEngine::new(128, 0).is_err());

        let def = PhaseSteeringEngine::default()
            .with_phase_clamp_max_rad(0.18)
            .with_phase_step_min_rad(0.01);
        assert_eq!(def.head_dim(), 128);
        assert_eq!(def.num_heads(), 32);
        assert!((def.phase_clamp_max_rad() - 0.18).abs() < 1e-6);
        assert!((def.phase_step_min_rad() - 0.01).abs() < 1e-6);
        assert_eq!(def.as_bytes().len(), 16);

        let eng_small = PhaseSteeringEngine::new(8, 2).unwrap();
        let mut q128 = AlignedBuffer128::new([
            1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
        ]);
        let p16 = AlignedBuffer16::new([64i8; 16]);
        eng_small
            .apply_phase_steering_aligned::<128>(
                &mut q128.as_aligned_slice_mut(),
                &p16.as_aligned_slice(),
            )
            .unwrap();
        assert!((q128.data[0] - 0.0).abs() < 1e-4);
        assert!((q128.data[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_qdps_filter_and_swar_scaling() {
        let engine = PhaseSteeringEngine::new(128, 16).unwrap();
        let mut raw_buf =
            AlignedBuffer16::new([1i8, 5, -2, 10, -128, 4, -1, 8, 6, -6, 12, -12, 0, 2, -3, 16]);
        let mut aligned = raw_buf.as_aligned_slice_mut();

        assert!(engine.phase_step_filter(&mut aligned, 0.1).is_ok());
        let res = aligned.as_slice();
        assert_eq!(res[0], 0);
        assert_eq!(res[1], 5);
        assert_eq!(res[2], 0);
        assert_eq!(res[3], 10);
        assert_eq!(res[4], -128);
        assert_eq!(res[5], 0);
        assert_eq!(res[6], 0);
        assert_eq!(res[7], 8);
        assert_eq!(res[15], 16);
    }

    #[test]
    fn test_lazy_unwrap_override() {
        let engine = PhaseSteeringEngine::new(4, 1).unwrap();
        let mut q_buf = AlignedBuffer32::new([1.0f32, 0.0, 0.0, 1.0]);
        let c_buf = AlignedBuffer32::new([PI * 0.5]);

        engine
            .lazy_unwrap_override(&mut q_buf.as_aligned_slice_mut(), &c_buf.as_aligned_slice())
            .unwrap();
        assert!((q_buf.data[0] - 0.0).abs() < 1e-4);
        assert!((q_buf.data[1] - 1.0).abs() < 1e-4);
        assert!((q_buf.data[2] - (-1.0)).abs() < 1e-4);
        assert!((q_buf.data[3] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_slerp_norm_guard_logit_shift_and_procrustes() {
        let engine = PhaseSteeringEngine::new(4, 2).unwrap();
        let mut t_buf = AlignedBuffer32::new([2.0f32; 8]);
        engine
            .slerp_norm_guard_stable(&mut t_buf.as_aligned_slice_mut(), 1.0, 1.0)
            .unwrap();
        let norm_sq: f32 = t_buf.data.iter().map(|v| v * v).sum();
        assert!((branchless_sqrt_f32(norm_sq) - 1.0).abs() < 1e-5);

        let mut logits = AlignedBuffer32::new([1.0f32; 8]);
        let mut phases_arr = [0i8; 16];
        phases_arr[0] = 32;
        phases_arr[1] = 64;
        let phases = AlignedBuffer16::new(phases_arr);
        engine
            .fused_logit_shift(
                &mut logits.as_aligned_slice_mut(),
                &phases.as_aligned_slice(),
            )
            .unwrap();
        let expected_shift = (48.0 * PI) / 128.0;
        for &val in &logits.data {
            assert!((val - (1.0 + expected_shift)).abs() < 1e-4);
        }

        let engine_8h = PhaseSteeringEngine::new(16, 8).unwrap();
        let mut src_8h = AlignedBuffer32::new([0.0f32; 128]);
        for pair in src_8h.data.chunks_exact_mut(2) {
            pair[0] = 1.0;
            pair[1] = 0.0;
        }
        let mut tgt_8h = src_8h;
        let p_64 = AlignedBuffer16::new([64i8; 16]);
        engine_8h
            .apply_phase_steering(&mut tgt_8h.as_aligned_slice_mut(), &p_64.as_aligned_slice())
            .unwrap();
        let mut out_8h = AlignedBuffer16::new([0i8; 16]);
        engine_8h
            .compute_procrustes_phase_delta(
                &src_8h.as_aligned_slice(),
                &tgt_8h.as_aligned_slice(),
                &mut out_8h.as_aligned_slice_mut(),
            )
            .unwrap();
        for h in 0..8 {
            assert_eq!(out_8h.data[h], 64);
        }

        let (s_pi, c_pi) = branchless_sincos_q7(-128);
        assert!(s_pi.abs() < 1e-6);
        assert!((c_pi - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_residual_adapter_engine_pod_patch_and_active_gate() {
        let mut engine = ResidualAdapterEngine::default();
        assert_eq!(engine.validate(), Ok(()));
        assert_eq!(
            engine.apply_patch(b"LiminiKa-Rank-1 residual-Patch"),
            Ok(())
        );
        assert!(engine.descriptor.scale_factor >= 0.25);

        let mut y_first = AlignedBuffer32::new([1.0_f32; 8]);
        let mut y_second = AlignedBuffer32::new([1.0_f32; 8]);
        let x = AlignedBuffer32::new([1.0_f32; 8]);
        let u = AlignedBuffer32::new([0.5_f32; 8]);
        let v = AlignedBuffer32::new([0.25_f32; 8]);

        let boundary_active =
            match Scoped::from_boundary(ResidualScale(1.0_f32)).cut_by_lacuna(true, 1.0, 0.0) {
                ScopedCut::Exterior(ext) => ext.into_active(),
                _ => panic!("expected exterior boundary active cut"),
            };
        engine
            .eval_rank1_slices_fused_active(
                &mut y_first.as_aligned_slice_mut(),
                &x.as_aligned_slice(),
                &u.as_aligned_slice(),
                &v.as_aligned_slice(),
                None,
                boundary_active,
            )
            .unwrap();
        let active_scale =
            match Scoped::from_boundary(ResidualScale(1.0_f32)).cut_by_lacuna_q7(true, 8, 4) {
                ScopedCut::Exterior(ext) => ext.into_active(),
                _ => panic!("expected exterior active cut"),
            };
        assert_eq!(active_scale.domain_id(), "stigmergy.residual_scale");
        engine
            .eval_rank1_slices_fused_active(
                &mut y_second.as_aligned_slice_mut(),
                &x.as_aligned_slice(),
                &u.as_aligned_slice(),
                &v.as_aligned_slice(),
                None,
                active_scale,
            )
            .unwrap();
        for (lhs, rhs) in y_first.data.iter().zip(y_second.data.iter()) {
            assert_eq!(lhs.to_bits(), rhs.to_bits());
        }
    }
}
