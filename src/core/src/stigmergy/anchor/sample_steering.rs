// name: src/core/src/stigmergy/anchor/sample_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Coherence-safe [`EpisodicPhaseSample`] and blanket [`EpisodicSampleSteering`].

use super::PhaseAnchorRecord;
use crate::foundation::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_min_f32,
    branchless_rad_to_phase_step_u8, branchless_relu_f32, branchless_splitmix64,
    branchless_von_mises_kappa, circular_r_bar_from_words_64, is_phase_profile_nonzero_64,
    is_word_array_uniform_64, load_q7_words_64, store_q7_word_at, store_q7_words_64,
    swar_q7_phase_step_filter_words_64, swar_q7_phase_step_filter_x8,
    swar_q7_scale_le128_with_bias_x8, swar_words_nonzero_64, AlignedBuffer16, AlignedSlice,
    AlignedSliceMut32, PhaseSteering, Pod, SWAR_LOW7_MASK_64, SWAR_SIGN_MASK_64,
    SWAR_SPLAT_BYTE_64,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_i32, LmnkResult, PhaseQ7, Scoped,
    LMNK_ERR_INVALID_ARGUMENT, MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS,
};
use crate::stigmergy::gate::{PhaseDeltaSlice, PhaseSteeringActiveExt};
use crate::stigmergy::trail::{trail_phase_words_ref, PointerTrail, TaggedPtr};
use crate::stigmergy::{
    ATTRACTOR_SLOT_KNOWLEDGE, ATTRACTOR_SLOT_PERSONA, MAX_ATTRACTOR_PULL_FORCE,
    MAX_STIGMERGIC_DENSITY, TRAIL_FLAG_REPULSION_ACTIVE,
};

// ===================================================================
// Internal Branchless & SWAR Phase Synthesis Helpers
// ===================================================================

/// Deterministically projects raw byte slices into a 64-head Q7 phase profile using
/// 8-byte SplitMix64 bijective hashing in $\mathcal{O}(L)$ with zero heap allocation.
#[inline]
#[must_use]
pub const fn hash_bytes_to_q7_phases(
    bytes: &[u8],
    domain_salt: u32,
) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
    let mut seed = 0xCBF2_9CE4_8422_2325_u64 ^ ((domain_salt as u64) << 32);
    let len = bytes.len();
    let mut i = 0usize;
    while i + 8 <= len {
        let word = (bytes[i] as u64)
            | ((bytes[i + 1] as u64) << 8)
            | ((bytes[i + 2] as u64) << 16)
            | ((bytes[i + 3] as u64) << 24)
            | ((bytes[i + 4] as u64) << 32)
            | ((bytes[i + 5] as u64) << 40)
            | ((bytes[i + 6] as u64) << 48)
            | ((bytes[i + 7] as u64) << 56);
        seed = branchless_splitmix64(seed ^ word.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        i += 8;
    }
    while i < len {
        seed = branchless_splitmix64(seed ^ (bytes[i] as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        i += 1;
    }
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    let mut w = 0usize;
    while w < 8 {
        seed = branchless_splitmix64(
            seed.wrapping_add((w as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
        );
        let shifted = (seed >> 1) & SWAR_LOW7_MASK_64;
        let signed_w = shifted ^ (seed & SWAR_SIGN_MASK_64);
        store_q7_word_at(&mut out, w * 8, signed_w);
        w += 1;
    }
    out
}

/// Deterministically projects a UTF-8 prompt string into a 64-head Q7 phase profile.
#[inline(always)]
#[must_use]
pub(crate) const fn hash_prompt_to_q7_phases(
    prompt: &str,
    domain_salt: u32,
) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
    hash_bytes_to_q7_phases(prompt.as_bytes(), domain_salt)
}

/// Deterministically quantizes a continuous `f32` embedding slice into a 64-head Q7 phase profile
/// in branchless `const fn` mode.
#[inline]
#[must_use]
pub const fn project_f32_slice_to_q7_phases(embedding: &[f32]) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
    let mut out = [0i8; MAX_ATTENTION_HEADS];
    let len = embedding.len();
    if len == 0 {
        return out;
    }
    let mut h = 0usize;
    if len >= MAX_ATTENTION_HEADS {
        while h < MAX_ATTENTION_HEADS {
            let sample_val = branchless_flush_subnormal_f32(embedding[h]);
            let q7_val = (sample_val * 128.0) as i32;
            out[h] = branchless_select_i32(
                q7_val < -128,
                -128,
                branchless_select_i32(q7_val > 127, 127, q7_val),
            ) as PhaseQ7;
            h += 1;
        }
    } else {
        while h < MAX_ATTENTION_HEADS {
            let sample_val = branchless_flush_subnormal_f32(embedding[h % len]);
            let q7_val = (sample_val * 128.0) as i32;
            out[h] = branchless_select_i32(
                q7_val < -128,
                -128,
                branchless_select_i32(q7_val > 127, 127, q7_val),
            ) as PhaseQ7;
            h += 1;
        }
    }
    out
}

// ===================================================================
// 1. Coherence-Safe Base Signal Trait (`EpisodicPhaseSample`)
// ===================================================================

/// Base signal trait representing any unknown input, prompt fragment, episodic memory turn,
/// continuous embedding, or latent cognitive signal that behaves as a sample point on the
/// unified $SO(2)^{64}$ attractor manifold.
///
/// # Coherence-Safe Blanket Architecture (`E0119` Prevention)
/// Instead of an unconstrained upstream `impl<T: AsRef<[u8]>>` (which collides with `[i8; 64]`
/// and tuples under Rust's orphan/coherence rules), `EpisodicPhaseSample` is implemented for
/// disjoint concrete head types (`str`, `[u8]`, `[u8; N]`, `[PhaseQ7; 64]`, `[f32]`, `[f32; N]`,
/// `AlignedSlice`, `PointerTrail`, `TaggedPtr`, `PhaseAnchorRecord`)
/// combined with blanket reference delegators (`&T`, `&mut T`) and
/// tuple decorators (`(S, f32)`, `(S, f32, u32)`).
pub trait EpisodicPhaseSample {
    /// Extracts or deterministically projects the sample's 64-head Q7 phase coordinates on $SO(2)^{64}$.
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS];

    /// Returns the initial mass / emotional or semantic salience of this sample point (`>= 0.0`).
    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        1.0
    }

    /// Returns the intrinsic phase consistency / coherence prior $c_s \in [0.0, 1.0]$ of this sample.
    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        1.0
    }

    /// Returns the preferred working lane (`0..=7`) for this sample point.
    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(ATTRACTOR_SLOT_PERSONA)
    }
}

macro_rules! impl_episodic_ref_delegator {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl<T: EpisodicPhaseSample + ?Sized> EpisodicPhaseSample for $ty {
                #[inline(always)]
                fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
                    (**self).sample_phase_coords(domain_salt)
                }
                #[inline(always)]
                fn sample_mass(&self) -> f32 {
                    (**self).sample_mass()
                }
                #[inline(always)]
                fn sample_coherence_prior(&self) -> f32 {
                    (**self).sample_coherence_prior()
                }
                #[inline(always)]
                fn preferred_working_lane(&self) -> u32 {
                    (**self).preferred_working_lane()
                }
                }
        )+
    };
}

impl_episodic_ref_delegator!(&T, &mut T);

macro_rules! impl_episodic_f32_seq {
    ($ty:ty $(; const $n:ident)? ; |$self_arg:ident| $as_slice:expr, $is_nonempty:expr) => {
        impl $(<const $n: usize>)? EpisodicPhaseSample for $ty {
            #[inline(always)]
            fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
                let $self_arg = self;
                project_f32_slice_to_q7_phases($as_slice)
            }
            #[inline(always)]
            fn sample_mass(&self) -> f32 {
                let $self_arg = self;
                let _ = $self_arg;
                branchless_select_f32($is_nonempty, 1.0, 0.0)
            }
            #[inline(always)]
            fn preferred_working_lane(&self) -> u32 {
                u32::from(ATTRACTOR_SLOT_KNOWLEDGE)
            }
        }
    };
}

impl EpisodicPhaseSample for str {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        hash_prompt_to_q7_phases(self, domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }
}

impl EpisodicPhaseSample for [u8] {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        hash_bytes_to_q7_phases(self, domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }
}

impl<const N: usize> EpisodicPhaseSample for [u8; N] {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        hash_bytes_to_q7_phases(self.as_slice(), domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(N > 0, 1.0, 0.0)
    }
}

impl EpisodicPhaseSample for [PhaseQ7; MAX_ATTENTION_HEADS] {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        *self
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(is_phase_profile_nonzero_64(self), 1.0, 0.0)
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        let r_bar = circular_r_bar_from_words_64(&load_q7_words_64(self));
        branchless_clamp_f32(r_bar, 0.25, 1.0)
    }
}

impl<const ALIGN: usize> EpisodicPhaseSample for AlignedSlice<'_, PhaseQ7, ALIGN> {
    #[inline]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        let mut out = [0i8; MAX_ATTENTION_HEADS];
        let s = self.as_slice();
        let len = s.len();
        if len == 0 {
            return out;
        }
        if len >= MAX_ATTENTION_HEADS {
            out.copy_from_slice(&s[..MAX_ATTENTION_HEADS]);
        } else {
            let mut h = 0usize;
            while h < MAX_ATTENTION_HEADS {
                out[h] = s[h % len];
                h += 1;
            }
        }
        out
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }
}

impl_episodic_f32_seq!([f32]; |s| s, !s.is_empty());
impl_episodic_f32_seq!([f32; N]; const N; |s| s.as_slice(), N > 0);
impl_episodic_f32_seq!(AlignedSlice<'_, f32, ALIGN>; const ALIGN; |s| s.as_slice(), !s.is_empty());

impl EpisodicPhaseSample for PointerTrail {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.accumulated_phase_delta
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        let rho = branchless_clamp_f32(self.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
        let pull = branchless_clamp_f32(self.attractor_pull_force, 0.25, MAX_ATTRACTOR_PULL_FORCE);
        let not_repulsed = (self.flags & TRAIL_FLAG_REPULSION_ACTIVE) == 0;
        branchless_select_f32(
            not_repulsed,
            branchless_flush_subnormal_f32(branchless_clamp_f32(
                0.5 * (rho + pull),
                0.1,
                MAX_ATTRACTOR_PULL_FORCE,
            )),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        let r_bar = circular_r_bar_from_words_64(trail_phase_words_ref(self));
        branchless_clamp_f32(r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(self.target_attractor_slot())
    }
}

impl EpisodicPhaseSample for TaggedPtr {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        let base_q7 = self.unpack_phase();
        let addr_bytes = (self.unpack_addr() ^ u64::from(domain_salt)).to_le_bytes();
        let mut out = hash_bytes_to_q7_phases(&addr_bytes, domain_salt);
        if base_q7 != 0 {
            out = [base_q7; MAX_ATTENTION_HEADS];
        }
        out
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32((!self.is_null()) | (self.unpack_phase() != 0), 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(self.target_attractor_slot())
    }
}

impl EpisodicPhaseSample for PhaseAnchorRecord {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.phase_profile
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(
            self.sample_count > 0,
            branchless_clamp_f32(
                branchless_flush_subnormal_f32(branchless_relu_f32(self.phase_gravity_mass)),
                0.1,
                MAX_ATTRACTOR_PULL_FORCE,
            ),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        branchless_clamp_f32(self.resultant_r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(self.dominant_slot) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }
}

impl<S: EpisodicPhaseSample> EpisodicPhaseSample for (S, f32) {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.0.sample_phase_coords(domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_flush_subnormal_f32(branchless_relu_f32(self.1))
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        self.0.sample_coherence_prior()
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.0.preferred_working_lane()
    }
}

impl<S: EpisodicPhaseSample> EpisodicPhaseSample for (S, f32, u32) {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        self.0.sample_phase_coords(domain_salt ^ self.2)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_flush_subnormal_f32(branchless_relu_f32(self.1))
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        self.0.sample_coherence_prior()
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.2 & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }
}

// ===================================================================
// 2. Blanket RoPE Direct Steering & Attractor Growth
// ===================================================================

/// 16-byte [`Pod`] telemetry report summarizing the personality growth and precision gain achieved
/// when absorbing one or more [`EpisodicPhaseSample`] points into the phase-anchor field.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AttractorGrowthReport {
    /// Resonant gravitational pull force exerted by the absorbed sample(s) on the active trail.
    pub resonant_pull_force: f32,
    /// Circular resultant vector length $\bar{R}_c \in [0, 1]$ measuring phase alignment precision.
    pub resultant_r_bar: f32,
    /// Von Mises concentration parameter $\kappa_c$ (higher values indicate sharper, more deterministic persona steering).
    pub von_mises_kappa: f32,
    /// Newly crystallized anchor ID (`> 0`) if the sample triggered bottom-up crystallization, or `0`.
    pub crystallized_anchor_id: u32,
}

// SAFETY: `AttractorGrowthReport` is a 16-byte `#[repr(C, align(16))]` struct of primitive `f32`/`u32` fields.
unsafe impl Pod for AttractorGrowthReport {}

/// Blanket-implemented behavioral trait (`impl<T: EpisodicPhaseSample + ?Sized> EpisodicSampleSteering for T {}`)
/// that automatically equips every [`EpisodicPhaseSample`] type with:
/// 1. Direct conversion to SWAR-scaled, `PhaseStepFilter`-filtered RoPE phase deltas (`[PhaseQ7; 64]`),
/// 2. Direct in-place `PhaseClamp` RoPE steering of Query/Key tensors via [`PhaseSteering`], and
/// 3. Constant-time $S^1$ geodesic resonance absorption into [`PointerTrail`].
pub trait EpisodicSampleSteering: EpisodicPhaseSample {
    /// Projects `self` onto $SO(2)^{64}$, scales all 64 heads by its bounded mass $\times$ coherence
    /// prior using 8-lane SWAR (`branchless_q7_scale_64`), and filters sub-grid phase jitter via
    /// 8-lane SWAR `PhaseStepFilter` (`branchless_q7_phase_step_filter_64`) in strictly constant time (`0` branches).
    #[inline]
    #[must_use]
    fn to_rope_phase_deltas(
        &self,
        domain_salt: u32,
        min_step_rad: f32,
    ) -> [PhaseQ7; MAX_ATTENTION_HEADS] {
        let mut phases = self.sample_phase_coords(domain_salt);
        let mass = branchless_clamp_f32(branchless_relu_f32(self.sample_mass()), 0.0, 2.0);
        let coh_prior =
            branchless_clamp_f32(branchless_relu_f32(self.sample_coherence_prior()), 0.0, 1.0);
        let eff_gain = branchless_clamp_f32(mass * coh_prior, 0.0, 1.0);
        let scale_u7 = (eff_gain * 128.0 + 0.5) as u8;
        if scale_u7 == 0 {
            return [0i8; MAX_ATTENTION_HEADS];
        }
        let step_u8 = branchless_rad_to_phase_step_u8(min_step_rad);
        let mut w = load_q7_words_64(&phases);
        if is_word_array_uniform_64(&w) {
            let mut w0 = w[0];
            if scale_u7 < 128 {
                let s = scale_u7 as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                w0 = swar_q7_scale_le128_with_bias_x8(w0, s, bias_restore);
            }
            if step_u8 > 1 {
                w0 = swar_q7_phase_step_filter_x8(w0, step_u8);
            }
            w = [w0; 8];
        } else {
            if scale_u7 < 128 {
                let s = scale_u7 as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                w = [
                    swar_q7_scale_le128_with_bias_x8(w[0], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[1], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[2], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[3], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[4], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[5], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[6], s, bias_restore),
                    swar_q7_scale_le128_with_bias_x8(w[7], s, bias_restore),
                ];
            }
            if step_u8 > 1 {
                w = swar_q7_phase_step_filter_words_64(&w, step_u8);
            }
        }
        store_q7_words_64(&mut phases, &w);
        phases
    }

    /// Directly steers `query_tensor` (and optional `key_tensor`) in-place via `PhaseClamp` soft-clamped
    /// RoPE phase rotation (`PhaseSteering`) using `self` as a behavioral phase sample, returning
    /// the applied 64-head Q7 phase delta vector.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `max_rad` is non-finite or `<= 0.0`, or
    /// propagates any error from [`PhaseSteeringActiveExt::apply_phase_steering_safe_active`] or
    /// [`PhaseSteeringActiveExt::apply_phase_steering_fused_safe_active`].
    #[inline]
    fn steer_rope_tensors_direct<P: PhaseSteering + ?Sized>(
        &self,
        engine: &P,
        domain_salt: u32,
        min_step_rad: f32,
        max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> LmnkResult<[PhaseQ7; MAX_ATTENTION_HEADS]> {
        if !max_rad.is_finite() || max_rad <= 0.0 {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let phases = self.to_rope_phase_deltas(domain_salt, min_step_rad);
        let aligned_phases = AlignedBuffer16::new(phases);
        let phase_slice = aligned_phases.as_aligned_slice();
        if let Some(active_deltas) = Scoped::from_boundary(PhaseDeltaSlice(phase_slice))
            .cut_by_lacuna(true, max_rad, 0.0)
            .into_active()
        {
            if let Some(k) = key_tensor {
                engine.apply_phase_steering_fused_safe_active(
                    query_tensor,
                    k,
                    active_deltas,
                    max_rad,
                )?;
            } else {
                engine.apply_phase_steering_safe_active(query_tensor, active_deltas, max_rad)?;
            }
        }
        Ok(phases)
    }

    /// Absorbs `self` directly into a substrate pointer trail (`trail_inout`) via $S^1$ geodesic
    /// phase-proximity resonance, updates `stigmergic_density`, evaluates circular statistics
    /// $(\bar{R}_c, \kappa_c)$, and stamps [`crate::stigmergy::TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`]
    /// in constant time (`0` data-dependent branches, `0 fdiv`).
    #[inline]
    fn resonate_into_trail(
        &self,
        domain_salt: u32,
        crystallize_density_thresh: f32,
        crystallize_r_bar_thresh: f32,
        trail_inout: &mut PointerTrail,
    ) -> AttractorGrowthReport {
        let coords = self.sample_phase_coords(domain_salt);
        let raw_mass = branchless_flush_subnormal_f32(branchless_relu_f32(self.sample_mass()));
        let prior = branchless_clamp_f32(
            branchless_relu_f32(self.sample_coherence_prior()),
            0.25,
            1.0,
        );
        let effective_mass = branchless_flush_subnormal_f32(raw_mass * prior);
        let coord_w = load_q7_words_64(&coords);
        let has_signal = (effective_mass > 1.0e-9) & swar_words_nonzero_64(&coord_w);
        let gated_mass = branchless_select_f32(has_signal, effective_mass, 0.0);

        let (resonant_pull, coherence) = if has_signal {
            trail_inout.apply_unified_manifold_anchor_resonance(&coords, gated_mass)
        } else {
            (0.0, 0.0)
        };
        let density_boost = branchless_flush_subnormal_f32(
            0.20 * gated_mass * (1.0 + branchless_relu_f32(coherence)),
        );
        trail_inout.stigmergic_density = branchless_min_f32(
            branchless_relu_f32(trail_inout.stigmergic_density) + density_boost,
            MAX_STIGMERGIC_DENSITY,
        );

        let r_bar = circular_r_bar_from_words_64(trail_phase_words_ref(trail_inout));
        let kappa = branchless_von_mises_kappa(r_bar);
        let _ = trail_inout.evaluate_crystallization_candidacy(
            crystallize_density_thresh,
            crystallize_r_bar_thresh,
            r_bar,
        );

        AttractorGrowthReport {
            resonant_pull_force: branchless_select_f32(has_signal, resonant_pull, 0.0),
            resultant_r_bar: r_bar,
            von_mises_kappa: kappa,
            crystallized_anchor_id: 0,
        }
    }
}

impl<T: EpisodicPhaseSample + ?Sized> EpisodicSampleSteering for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{
        is_phase_profile_nonzero_64, AlignedBuffer32, PhaseSteering, DEFAULT_PHASE_CLAMP_MAX_RAD,
    };
    use crate::liminal::{PhaseQ7, MAX_ATTENTION_HEADS};

    struct DummyPhaseSteerer;
    impl PhaseSteering for DummyPhaseSteerer {
        fn head_dim(&self) -> u32 {
            8
        }
        fn num_heads(&self) -> u32 {
            2
        }
    }

    #[test]
    fn test_episodic_sample_steering_blanket_projections() {
        let steerer = DummyPhaseSteerer;
        let mut q_tensor = AlignedBuffer32::new([1.0f32; 16]);
        let str_deltas = "persistent persona anchor sample"
            .steer_rope_tensors_direct(
                &steerer,
                1,
                0.01,
                DEFAULT_PHASE_CLAMP_MAX_RAD,
                &mut q_tensor.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert!(is_phase_profile_nonzero_64(&str_deltas));

        let raw_q7_sample: [PhaseQ7; MAX_ATTENTION_HEADS] = [32i8; MAX_ATTENTION_HEADS];
        let q7_deltas = raw_q7_sample.to_rope_phase_deltas(1, 0.01);
        assert_eq!(q7_deltas[0], 32);

        let f32_sample = [0.25f32; 16];
        let f32_deltas = f32_sample.to_rope_phase_deltas(2, 0.01);
        assert_eq!(f32_deltas[0], 32);
    }
}
