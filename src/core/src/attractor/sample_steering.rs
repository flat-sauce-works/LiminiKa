// name: src/core/src/attractor/sample_steering.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Episodic phase sample trait, blanket RoPE steering, and attractor growth coordinator.

use super::*;

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
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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
pub(super) const fn hash_prompt_to_q7_phases(
    prompt: &str,
    domain_salt: u32,
) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
    hash_bytes_to_q7_phases(prompt.as_bytes(), domain_salt)
}

/// Deterministically quantizes a continuous `f32` embedding slice into a 64-head Q7 phase profile
/// in branchless `const fn` mode.
#[inline]
#[must_use]
pub const fn project_f32_slice_to_q7_phases(embedding: &[f32]) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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
            ) as gcso_q7_t;
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
            ) as gcso_q7_t;
            h += 1;
        }
    }
    out
}

/// Synthesizes a 16-element normalized `f32` embedding preview from a 64-head Q7 phase array
/// so that absorbed phase samples can be registered via [`AttractorField::add_embedding_anchor`].
#[inline(always)]
#[must_use]
pub(super) const fn q7_phases_to_embedding16(
    phases: &[gcso_q7_t; MAX_ATTENTION_HEADS],
) -> [f32; 16] {
    let mut emb = [0.0_f32; 16];
    let mut i = 0usize;
    while i < 16 {
        emb[i] = (phases[i] as f32) * (1.0 / 128.0);
        i += 1;
    }
    emb
}

// ===================================================================
// 1. Pillar 1: Coherence-Safe Base Signal Trait (`EpisodicPhaseSample`)
// ===================================================================

/// Base signal trait representing any unknown input, prompt fragment, episodic memory turn,
/// continuous embedding, or latent cognitive signal that behaves as a sample point on the
/// unified $SO(2)^{64}$ attractor manifold.
///
/// # Coherence-Safe Blanket Architecture (`E0119` Prevention)
/// Instead of an unconstrained upstream `impl<T: AsRef<[u8]>>` (which collides with `[i8; 64]`
/// and tuples under Rust's orphan/coherence rules), `EpisodicPhaseSample` is implemented for
/// disjoint concrete head types (`str`, `[u8]`, `[u8; N]`, `[gcso_q7_t; 64]`, `[f32]`, `[f32; N]`,
/// `AlignedSlice`, `gcso_pointer_trail_t`, `gcso_tagged_ptr_t`, `AttractorAnchor`,
/// `CrystallizedAnchorRecord`) combined with blanket reference delegators (`&T`, `&mut T`) and
/// tuple decorators (`(S, f32)`, `(S, f32, u32)`).
pub trait EpisodicPhaseSample {
    /// Extracts or deterministically projects the sample's 64-head Q7 phase coordinates on $SO(2)^{64}$.
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS];

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
        DOMAIN_PERSONA
    }

    /// Returns the C-ABI anchor classification type when this sample crystallizes into a permanent anchor.
    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_CRYSTALLIZED
    }
}

impl<T: EpisodicPhaseSample + ?Sized> EpisodicPhaseSample for &T {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        (**self).sample_anchor_type()
    }
}

impl<T: EpisodicPhaseSample + ?Sized> EpisodicPhaseSample for &mut T {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        (**self).sample_anchor_type()
    }
}

impl EpisodicPhaseSample for str {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        hash_prompt_to_q7_phases(self, domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_PERSONA
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_SYSTEM_PROMPT
    }
}

impl EpisodicPhaseSample for [u8] {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        hash_bytes_to_q7_phases(self, domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_PERSONA
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_SYSTEM_PROMPT
    }
}

impl<const N: usize> EpisodicPhaseSample for [u8; N] {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        hash_bytes_to_q7_phases(self.as_slice(), domain_salt)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(N > 0, 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_PERSONA
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_SYSTEM_PROMPT
    }
}

impl EpisodicPhaseSample for [gcso_q7_t; MAX_ATTENTION_HEADS] {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_PERSONA
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_TOPOLOGICAL
    }
}

impl<const ALIGN: usize> EpisodicPhaseSample for AlignedSlice<'_, gcso_q7_t, ALIGN> {
    #[inline]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_PERSONA
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_TOPOLOGICAL
    }
}

impl EpisodicPhaseSample for [f32] {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_phases(self)
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_KNOWLEDGE
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_EMBEDDING
    }
}

impl<const N: usize> EpisodicPhaseSample for [f32; N] {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_phases(self.as_slice())
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(N > 0, 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_KNOWLEDGE
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_EMBEDDING
    }
}

impl<const ALIGN: usize> EpisodicPhaseSample for AlignedSlice<'_, f32, ALIGN> {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        project_f32_slice_to_q7_phases(self.as_slice())
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(!self.is_empty(), 1.0, 0.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        DOMAIN_KNOWLEDGE
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_EMBEDDING
    }
}

impl EpisodicPhaseSample for gcso_pointer_trail_t {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_CRYSTALLIZED
    }
}

impl EpisodicPhaseSample for gcso_tagged_ptr_t {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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
    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_TOPOLOGICAL
    }
}

impl EpisodicPhaseSample for GatewayFrame {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_clamp_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(self.effective_energy)),
            0.0,
            MAX_ATTRACTOR_PULL_FORCE,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        branchless_clamp_f32(0.5 * (1.0 + self.fact_persona_coherence), 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        u32::from(self.dominant_slot) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_CRYSTALLIZED
    }
}

impl EpisodicPhaseSample for CrystallizedAttractorRecord {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        GCSO_ANCHOR_TYPE_CRYSTALLIZED
    }
}

impl<S: EpisodicPhaseSample> EpisodicPhaseSample for (S, f32) {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.0.sample_anchor_type()
    }
}

impl<S: EpisodicPhaseSample> EpisodicPhaseSample for (S, f32, u32) {
    #[inline(always)]
    fn sample_phase_coords(&self, domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
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

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.0.sample_anchor_type()
    }
}

// ===================================================================
// 2. Pillar 2: Blanket RoPE Direct Steering & Attractor Growth
// ===================================================================

/// 16-byte [`Pod`] telemetry report summarizing the personality growth and precision gain achieved
/// when absorbing one or more [`EpisodicPhaseSample`] points into the attractor field.
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
/// 1. Direct conversion to SWAR-scaled, QDPS-filtered RoPE phase deltas (`[gcso_q7_t; 64]`),
/// 2. Direct in-place RIPA RoPE steering of Query/Key tensors via [`PhaseSteering`],
/// 3. Constant-time $S^1$ geodesic resonance absorption into [`gcso_pointer_trail_t`], and
/// 4. Direct crystallization into a [`Pod`] [`AttractorAnchor`].
pub trait EpisodicSampleSteering: EpisodicPhaseSample {
    /// Projects `self` onto $SO(2)^{64}$, scales all 64 heads by its bounded mass $\times$ coherence
    /// prior using 8-lane SWAR (`branchless_q7_scale_64`), and filters sub-grid phase jitter via
    /// 8-lane SWAR QDPS (`branchless_q7_qdps_filter_64`) in strictly constant time (`0` branches).
    #[inline]
    #[must_use]
    fn to_rope_phase_deltas(
        &self,
        domain_salt: u32,
        qdps_min_step_rad: f32,
    ) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        let mut phases = self.sample_phase_coords(domain_salt);
        let mass = branchless_clamp_f32(branchless_relu_f32(self.sample_mass()), 0.0, 2.0);
        let coh_prior =
            branchless_clamp_f32(branchless_relu_f32(self.sample_coherence_prior()), 0.0, 1.0);
        let eff_gain = branchless_clamp_f32(mass * coh_prior, 0.0, 1.0);
        let scale_u7 = (eff_gain * 128.0 + 0.5) as u8;
        if scale_u7 == 0 {
            return [0i8; MAX_ATTENTION_HEADS];
        }
        let step_u8 = branchless_rad_to_qdps_step_u8(qdps_min_step_rad);
        let mut w = load_q7_words_64(&phases);
        if is_word_array_uniform_64(&w) {
            let mut w0 = w[0];
            if scale_u7 < 128 {
                let s = scale_u7 as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(SWAR_SPLAT_BYTE_64);
                w0 = swar_q7_scale_le128_with_bias_x8(w0, s, bias_restore);
            }
            if step_u8 > 1 {
                w0 = crate::traits::primitives::swar_q7_qdps_filter_x8(w0, step_u8);
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
                w = swar_q7_qdps_filter_words_64(&w, step_u8);
            }
        }
        store_q7_words_64(&mut phases, &w);
        phases
    }

    /// Directly steers `query_tensor` (and optional `key_tensor`) in-place via RIPA soft-clamped
    /// RoPE phase rotation (`PhaseSteering`) using `self` as a behavioral phase sample, returning
    /// the applied 64-head Q7 phase delta vector.
    ///
    /// # Errors
    /// Propagates any error from [`PhaseSteering::apply_phase_steering_safe`] or
    /// [`PhaseSteering::apply_phase_steering_fused_safe`].
    #[inline]
    fn steer_rope_tensors_direct<P: PhaseSteering + ?Sized>(
        &self,
        engine: &P,
        domain_salt: u32,
        qdps_min_step_rad: f32,
        ripa_max_rad: f32,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
    ) -> GcsoResult<[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        let phases = self.to_rope_phase_deltas(domain_salt, qdps_min_step_rad);
        let aligned_phases = AlignedBuffer16::new(phases);
        let phase_slice = aligned_phases.as_aligned_slice();
        if let Some(k) = key_tensor {
            engine.apply_phase_steering_fused_safe(query_tensor, k, &phase_slice, ripa_max_rad)?;
        } else {
            engine.apply_phase_steering_safe(query_tensor, &phase_slice, ripa_max_rad)?;
        }
        Ok(phases)
    }

    /// Absorbs `self` directly into an ActionHub pointer trail (`trail_inout`) via $S^1$ geodesic
    /// phase-proximity resonance, updates `stigmergic_density`, evaluates circular statistics
    /// $(\bar{R}_c, \kappa_c)$, and stamps [`crate::traits::telemetry::TRAIL_FLAG_CRYSTALLIZED_CANDIDATE`]
    /// in constant time (`0` data-dependent branches, `0 fdiv`).
    #[inline]
    fn resonate_into_trail(
        &self,
        domain_salt: u32,
        crystallize_density_thresh: f32,
        crystallize_r_bar_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
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
            trail_inout.apply_unified_field_anchor_resonance(&coords, gated_mass)
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

    /// Converts `self` directly into a 640-byte [`Pod`] [`AttractorAnchor`] ready for insertion
    /// into the unified manifold or `.gcso` persistence.
    #[inline]
    #[must_use]
    fn crystallize_into_anchor(
        &self,
        anchor_id: u32,
        current_step: u64,
        domain_salt: u32,
    ) -> AttractorAnchor {
        let mut phases = self.sample_phase_coords(domain_salt);
        let anchor_type = self.sample_anchor_type();
        if anchor_type == GCSO_ANCHOR_TYPE_PHASE_REPULSE {
            phases = branchless_q7_conjugate_64(&phases);
        }
        let mass = branchless_max_f32(
            branchless_flush_subnormal_f32(branchless_relu_f32(self.sample_mass())),
            0.1,
        );
        let mut vector_data = [0.0_f32; MAX_EMBEDDING_DIM];
        let mut i = 0usize;
        while i < 16 {
            vector_data[i] = (phases[i] as f32) * (1.0 / 128.0);
            i += 1;
        }
        AttractorAnchor {
            id: anchor_id,
            anchor_type,
            weight: mass,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            half_life_steps: 1000,
            last_active_step: current_step,
            domain_id: self.preferred_working_lane() & ((MAX_ATTRACTOR_DOMAINS as u32) - 1),
            vector_dim: 16,
            is_active: 1,
            is_crystallized: u8::from(anchor_type == GCSO_ANCHOR_TYPE_CRYSTALLIZED),
            reserved_padding: [0; 24],
            phase_deltas: phases,
            vector_data,
        }
    }
}

impl<T: EpisodicPhaseSample + ?Sized> EpisodicSampleSteering for T {}

/// Blanket-implemented coordinator trait (`impl<T: AttractorField + PersonaPatcher + ?Sized> AttractorGrowthCoordinator for T {}`)
/// that enables any attractor field or macro orchestrator to absorb arbitrary [`EpisodicPhaseSample`]
/// inputs as points on the $SO(2)^{64}$ manifold, strengthen gravitational precision ($\kappa_c$),
/// steer RoPE phase deltas + inverse temperature $\beta_{\mathrm{eff}}$, and crystallize persistent
/// persona modes ready for `.gcso` serialization.
pub trait AttractorGrowthCoordinator: AttractorField + PersonaPatcher {
    /// Absorbs a single [`EpisodicPhaseSample`] point onto the unified phase manifold:
    /// 1. Resonates `sample` into `trail_inout` via [`EpisodicSampleSteering::resonate_into_trail`].
    /// 2. Evaluates the updated 64-head circular resultant length $\bar{R}_c$ and von Mises
    ///    concentration $\kappa_c$.
    /// 3. If `trail_inout` crosses `(crystallize_density_thresh, crystallize_r_bar_thresh)`,
    ///    registers a new permanent anchor into `self` and triggers [`AttractorField::aggregate_bottom_up`].
    ///
    /// # Errors
    /// Propagates any error from [`AttractorField::add_embedding_anchor`] or [`AttractorField::aggregate_bottom_up`].
    #[inline]
    fn absorb_episodic_sample<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        domain_salt: u32,
        crystallize_density_thresh: f32,
        crystallize_r_bar_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
    ) -> GcsoResult<AttractorGrowthReport> {
        let mut report = sample.resonate_into_trail(
            domain_salt,
            crystallize_density_thresh,
            crystallize_r_bar_thresh,
            trail_inout,
        );
        if report.resonant_pull_force <= 0.0 {
            return Ok(report);
        }

        let should_crystallize =
            (trail_inout.flags & crate::traits::telemetry::TRAIL_FLAG_CRYSTALLIZED_CANDIDATE) != 0;
        if should_crystallize {
            let emb16 = AlignedBuffer32::new(q7_phases_to_embedding16(
                &trail_inout.accumulated_phase_delta,
            ));
            let mass = branchless_max_f32(
                branchless_flush_subnormal_f32(branchless_relu_f32(sample.sample_mass())),
                1.0,
            );
            if let Ok(new_id) = self.add_embedding_anchor(
                &emb16.as_aligned_slice(),
                sample.sample_anchor_type(),
                mass,
            ) {
                report.crystallized_anchor_id = new_id;
                trail_inout.target_anchor_id = new_id;
                trail_inout.cluster_id = new_id;
            }
            let _ = self.aggregate_bottom_up()?;
        }

        Ok(report)
    }

    /// Absorbs a sequence of [`EpisodicPhaseSample`] points across a conversation horizon,
    /// accumulating gravitational pull and sharpening persona concentration $\kappa_c$, and
    /// quantizes the resulting persistent topological persona profile into `persona_profile_out`.
    ///
    /// Returns `(final_growth_report, crystallized_anchor_count)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::absorb_episodic_sample`] or
    /// [`PersonaPatcher::quantize_persistent_topological_mode`].
    #[inline]
    fn absorb_episodic_sample_batch<S: EpisodicPhaseSample>(
        &mut self,
        samples: &[S],
        domain_salt: u32,
        crystallize_density_thresh: f32,
        crystallize_r_bar_thresh: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        persona_profile_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(AttractorGrowthReport, u32)> {
        let mut last_report = AttractorGrowthReport::default();
        let mut crystallized_total = 0u32;

        for (idx, sample) in samples.iter().enumerate() {
            let step_salt = domain_salt.wrapping_add(idx as u32);
            let rep = self.absorb_episodic_sample(
                sample,
                step_salt,
                crystallize_density_thresh,
                crystallize_r_bar_thresh,
                trail_inout,
            )?;
            if rep.crystallized_anchor_id != 0 {
                crystallized_total = crystallized_total.wrapping_add(1);
            }
            last_report = rep;
        }

        let trails = [*trail_inout];
        let (r_bar, kappa) = self.quantize_persistent_topological_mode(
            &trails,
            branchless_clamp_f32(crystallize_r_bar_thresh, 0.0, 1.0),
            persona_profile_out,
        )?;
        last_report.resultant_r_bar = r_bar;
        last_report.von_mises_kappa = kappa;
        Ok((last_report, crystallized_total))
    }

    /// Evaluates the precision-guided inverse-temperature scale $\beta_{\mathrm{eff}} \in [0.25, 4.0]$
    /// directly from a growth report's concentration $\kappa_c$ and phase coherence, allowing
    /// well-grown attractors (large sample count $\to$ high $\kappa_c$) to steer RoPE/Attention
    /// with higher precision than sparse initial attractors.
    #[inline(always)]
    #[must_use]
    fn eval_growth_precision_inv_temp(
        &self,
        growth_report: &AttractorGrowthReport,
        phase_coherence: f32,
        base_gain: f32,
    ) -> f32 {
        let kappa_norm = branchless_clamp_f32(growth_report.von_mises_kappa * 0.10, 0.25, 2.0);
        let eff_gain = branchless_relu_f32(base_gain) * kappa_norm;
        branchless_phase_coherence_inv_temp_scale(phase_coherence, eff_gain)
    }

    /// End-to-end blanket step that absorbs an [`EpisodicPhaseSample`] into `trail_inout`,
    /// evaluates unified-field gravitational force to produce a QDPS-filtered RoPE phase delta
    /// vector (`rope_deltas_out: &mut [gcso_q7_t; 64]`), and returns `(growth_report, inv_temp_beta_eff)`.
    ///
    /// # Errors
    /// Propagates any error from [`Self::absorb_episodic_sample`] or [`AttractorField::evaluate_field_force`].
    #[inline]
    fn absorb_and_steer_rope_step<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        domain_salt: u32,
        crystallize_density_thresh: f32,
        crystallize_r_bar_thresh: f32,
        qdps_min_step_rad: f32,
        inv_temp_base_gain: f32,
        trail_inout: &mut gcso_pointer_trail_t,
        rope_deltas_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<(AttractorGrowthReport, f32)> {
        let report = self.absorb_episodic_sample(
            sample,
            domain_salt,
            crystallize_density_thresh,
            crystallize_r_bar_thresh,
            trail_inout,
        )?;
        let _field_pull = self.evaluate_field_force(trail_inout, rope_deltas_out)?;
        let step_u8 = branchless_rad_to_qdps_step_u8(qdps_min_step_rad);
        let mut rope_w = load_q7_words_64(rope_deltas_out);
        if step_u8 > 1 {
            rope_w = swar_q7_qdps_filter_words_64(&rope_w, step_u8);
            store_q7_words_64(rope_deltas_out, &rope_w);
        }
        let coh = coherence_from_word_arrays_64(trail_phase_words_ref(trail_inout), &rope_w);
        let beta_eff = self.eval_growth_precision_inv_temp(&report, coh, inv_temp_base_gain);
        Ok((report, beta_eff))
    }
}

impl<T: AttractorField + PersonaPatcher + ?Sized> AttractorGrowthCoordinator for T {}
