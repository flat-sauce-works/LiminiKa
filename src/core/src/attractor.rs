// name: src/core/src/attractor.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unified Macro Attractor Field Engine, Dynamic 8-Lane Working Space, Blanket-Implemented
//! Episodic Sample -> RoPE Steering / Persona Growth Coordinator, and `.gcso` `Pod` Snapshot
//! Persistence for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! # Architectural Role (Unified Phase Manifold, `0-fdiv` $S^1$ Geodesic Gravity & `.gcso` Growth)
//! Rather than partitioning latent memory into rigid, manually engineered feature boxes,
//! this module models all persona traits, world lore, episodic dialogue turns, and anti-hallucination
//! repulsion traces as sample points (anchors) coexisting on a **single continuous phase manifold**
//! ($SO(2)^{d_{\mathrm{head}}/2}$, represented as 64-byte Q7 phase vectors):
//!
//! 1. **Coherence-Safe Base Signal Trait (`EpisodicPhaseSample`)**:
//!    Any cognitive input (`str`, `&str`, `[u8]`, `&[u8]`, `[u8; N]`, `[gcso_q7_t; 64]`, `[f32]`,
//!    `[f32; N]`, `AlignedSlice`, `gcso_pointer_trail_t`, `gcso_tagged_ptr_t`, `AttractorAnchor`,
//!    `CrystallizedAnchorRecord`, and weighted tuples `(S, f32)` / `(S, f32, u32)`) implements
//!    [`EpisodicPhaseSample`] via strictly disjoint head types and reference/tuple blanket delegators,
//!    completely avoiding upstream `AsRef<[u8]>` coherence collisions (`E0119`).
//! 2. **Blanket-Implemented RoPE Steering & Attractor Growth (`EpisodicSampleSteering` & `AttractorGrowthCoordinator`)**:
//!    - [`EpisodicSampleSteering`] (`impl<T: EpisodicPhaseSample + ?Sized>`) grants every sample type
//!      automatic conversion to SWAR-scaled, QDPS-filtered RoPE phase deltas (`[gcso_q7_t; 64]`),
//!      direct in-place Query/Key tensor RIPA phase steering, constant-time resonance absorption into
//!      [`gcso_pointer_trail_t`], and direct crystallization into [`AttractorAnchor`].
//!    - [`AttractorGrowthCoordinator`] (`impl<T: AttractorField + PersonaPatcher + ?Sized>`) plots incoming
//!      samples onto the manifold. As sample count grows, circular resultant length $\bar{R}_c \to 1$
//!      and von Mises concentration $\kappa_c$ rise, sharpening gravitational pull and inverse-temperature
//!      scaling $\beta_{\mathrm{eff}}$ before bottom-up crystallizing into persistent anchors.
//! 3. **Zero-Copy `.gcso` `Pod` Snapshot Persistence (`CrystallizedAnchorRecord` & `AttractorGrowthSnapshot`)**:
//!    Crystallized sample points, synthesized persona phase profiles, and concentration $\kappa_c$
//!    are packed into an 8,448-byte (8.25 KB) 64-byte-aligned [`Pod`] snapshot (`AttractorGrowthSnapshot`)
//!    with branchless IEEE 802.3 CRC32 verification for zero-copy ZIMMS `.gcso` distribution.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::inline_always)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_option_as_deref)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_slice_size_calculation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::ifs_same_cond)]
#![allow(clippy::field_reassign_with_default)]

use core::mem::{align_of, offset_of, size_of};

use crate::abi::{
    gcso_anchor_type_t, gcso_pointer_trail_t, gcso_q7_t, gcso_tagged_ptr_t, GCSO_ABI_VERSION_HEX,
    GCSO_ANCHOR_TYPE_CRYSTALLIZED, GCSO_ANCHOR_TYPE_EMBEDDING, GCSO_ANCHOR_TYPE_PHASE_REPULSE,
    GCSO_ANCHOR_TYPE_SYSTEM_PROMPT, GCSO_ANCHOR_TYPE_TOPOLOGICAL, GCSO_ERROR_ACTION_HUB_FULL,
    GCSO_ERROR_CONTAINER_CORRUPTED, GCSO_ERROR_INVALID_ARGUMENT, GCSO_ERROR_VERSION_MISMATCH,
};
use crate::traits::aligned::{
    AlignedBuffer16, AlignedBuffer32, AlignedSlice, AlignedSlice16, AlignedSlice32,
    AlignedSliceMut32,
};
use crate::traits::branchless::{
    branchless_clamp_f32, branchless_flush_subnormal_f32, branchless_inv_pos_f32,
    branchless_is_finite_f32, branchless_max_f32, branchless_min_f32,
    branchless_phase_coherence_inv_temp_scale, branchless_q7_conjugate_64,
    branchless_q7_inhibit_words_64, branchless_q7_resonant_pull_64,
    branchless_q7_synergy_fuse_words_64, branchless_rad_to_qdps_step_u8, branchless_relu_f32,
    branchless_select_f32, branchless_select_i32, branchless_select_u32, branchless_select_u8,
    branchless_splitmix64, branchless_von_mises_kappa, circular_r_bar_from_words_64,
    circular_resultant_from_words_64, coherence_from_word_arrays_64,
    coherence_from_word_arrays_with_both_uniform_flags_64, crc32_block64_zero_seed_words,
    crc32_step_block64_words, crc32_step_l16, crc32_step_l32, crc32_step_l8, fast_inv_len_f32,
    fast_inv_pos_normal_f32, is_phase_profile_nonzero_64, is_word_array_uniform_64,
    load_q7_words_64, store_q7_word_at, store_q7_words_64, swar_q7_qdps_filter_words_64,
    swar_q7_scale_le128_with_bias_x8, swar_q7_sub_sat_x8, trail_phase_words_ref, Q7_FULL_COS_REF,
};
use crate::traits::cold_path::{
    AttractorField, ColdPathTrailCrystalExt, MultiDomainAttractorSpace, PersonaPatcher,
};
use crate::traits::gateway::{
    CausalProvenanceSignal, CognitiveGroundingSignal, CrystallizedAttractorRecord, GatewayFrame,
    ManifoldGatewaySignal, SubjectivePersonaSignal, CAUSAL_AXIS_SUBJECTIVE_PERSONA,
    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED, ROOT_CAUSE_NOMINAL,
};
use crate::traits::primitives::{
    is_slot_pair_coupled, GcsoResult, Pod, DEFAULT_SLOT_COUPLING_MATRIX_8X8,
    GCSO_SNAPSHOT_MAGIC_U32, MAX_ATTENTION_HEADS,
};
use crate::traits::rope::PhaseSteering;
use crate::traits::telemetry::{
    StigmergicActionPtrOps, StigmergicTrailOps, MAX_ATTRACTOR_PULL_FORCE, MAX_STIGMERGIC_DENSITY,
    TRAIL_FLAG_REPULSION_ACTIVE,
};

pub use crate::traits::primitives::{MAX_ANCHORS_PER_FIELD, MAX_ATTRACTOR_DOMAINS};

/// Maximum dimension supported for embedding vectors in inline zero-alloc storage (`128`).
pub const MAX_EMBEDDING_DIM: usize = 128;

/// Default attractor lane identifier for System Baseline Constraints (`0`).
pub const DOMAIN_SYSTEM_BASELINE: u32 = 0;
/// Default attractor lane identifier for Persona Patches (`1`).
pub const DOMAIN_PERSONA: u32 = 1;
/// Default attractor lane identifier for Generalized Knowledge Sources (`2`).
pub const DOMAIN_KNOWLEDGE: u32 = 2;
/// Default attractor lane identifier for Anti-Hallucination Phase Repulsion (`3`).
pub const DOMAIN_PHASE_REPULSION: u32 = 3;
/// Default attractor lane identifier for User Interaction Context (`4`).
pub const DOMAIN_USER_INTERACTION: u32 = 4;
/// Dynamic extension working lane identifier 5 (`Visual / Screen Co-Watching`).
pub const DOMAIN_CUSTOM_5: u32 = 5;
/// Dynamic extension working lane identifier 6 (`Mood / Circadian Drift`).
pub const DOMAIN_CUSTOM_6: u32 = 6;
/// Dynamic extension working lane identifier 7 (`Workshop Extension / Overflow`).
pub const DOMAIN_CUSTOM_7: u32 = 7;

const SNAPSHOT_HEADER_BYTES: usize = 64;

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
        let shifted = (seed >> 1) & 0x7F7F_7F7F_7F7F_7F7F;
        let signed_w = shifted ^ (seed & 0x8080_8080_8080_8080);
        store_q7_word_at(&mut out, w * 8, signed_w);
        w += 1;
    }
    out
}

/// Deterministically projects a UTF-8 prompt string into a 64-head Q7 phase profile.
#[inline(always)]
#[must_use]
const fn hash_prompt_to_q7_phases(
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
const fn q7_phases_to_embedding16(phases: &[gcso_q7_t; MAX_ATTENTION_HEADS]) -> [f32; 16] {
    let mut emb = [0.0_f32; 16];
    let mut i = 0usize;
    while i < 16 {
        emb[i] = (phases[i] as f32) * (1.0 / 128.0);
        i += 1;
    }
    emb
}

#[inline(always)]
#[must_use]
const fn words_any_nonzero_64(w: &[u64; 8]) -> bool {
    if w[0] != 0 {
        return true;
    }
    ((w[1] | (w[2] | w[3])) | ((w[4] | w[5]) | (w[6] | w[7]))) != 0
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
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(0x0101_0101_0101_0101);
                w0 = swar_q7_scale_le128_with_bias_x8(w0, s, bias_restore);
            }
            if step_u8 > 1 {
                w0 = crate::traits::branchless::swar_q7_qdps_filter_x8(w0, step_u8);
            }
            w = [w0; 8];
        } else {
            if scale_u7 < 128 {
                let s = scale_u7 as u64;
                let bias_restore = (128u64.wrapping_sub(s)).wrapping_mul(0x0101_0101_0101_0101);
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
        let has_signal = (effective_mass > 1.0e-9) & words_any_nonzero_64(&coord_w);
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

// ===================================================================
// 3. Pillar 3: Compact `.gcso` Pod Snapshot Structures (`8.25 KB`)
// ===================================================================

/// Compact 128-byte (2-cacheline) [`Pod`] record storing a single crystallized or active sample
/// point on the $SO(2)^{64}$ manifold inside a `.gcso` snapshot (`AttractorGrowthSnapshot`).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrystallizedAnchorRecord {
    /// Target phase coordinates across 64 attention heads (Q7 fixed-point, 64 bytes, offset 0..64).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Unique anchor identifier (`1`-based, `0` = empty slot, offset 64..68).
    pub id: u32,
    /// Classification type of topological anchor (`gcso_anchor_type_t`, offset 68..72).
    pub anchor_type: gcso_anchor_type_t,
    /// Originating working lane identifier (`0..=7`, offset 72..76).
    pub domain_id: u32,
    /// Half-life in token steps for temporal decay (offset 76..80).
    pub half_life_steps: u32,
    /// Intrinsic mass / baseline salience (offset 80..84).
    pub weight: f32,
    /// Dynamic gravitational pull force $F_{\mathrm{pull}}$ (offset 84..88).
    pub pull_force: f32,
    /// Accumulated stigmergic trail density $\rho_{\mathrm{stigmergy}}$ (offset 88..92).
    pub stigmergic_density: f32,
    /// Circular resultant vector length $\bar{R}_c \in [0, 1]$ of this sample's phase profile (offset 92..96).
    pub resultant_r_bar: f32,
    /// Von Mises concentration parameter $\kappa_c$ of this sample's phase profile (offset 96..100).
    pub von_mises_kappa: f32,
    /// Active flag (`1` = active, `0` = inactive, offset 100).
    pub is_active: u8,
    /// Crystallized flag (`1` = bottom-up crystallized, `0` = transient, offset 101).
    pub is_crystallized: u8,
    /// Explicit 2-byte alignment padding before 8-byte `last_active_step` (offset 102..104).
    pub reserved_flags: u16,
    /// Token step index when this anchor was last activated (offset 104..112).
    pub last_active_step: u64,
    /// Explicit trailing padding to 128-byte (2-cacheline) boundary (offset 112..128).
    pub reserved_padding: [u8; 16],
}

// SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` of size 128 with zero implicit padding traps.
unsafe impl Pod for CrystallizedAnchorRecord {}

impl Default for CrystallizedAnchorRecord {
    #[inline(always)]
    fn default() -> Self {
        Self::zeroed()
    }
}

impl EpisodicPhaseSample for CrystallizedAnchorRecord {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(
            self.is_active != 0,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        branchless_clamp_f32(self.resultant_r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.anchor_type
    }
}

impl CrystallizedAnchorRecord {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 0..64`,
    /// guaranteed 64-byte aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` and `phase_deltas` is at offset 0
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &*(self
                .phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably borrows `self.phase_deltas` (`offset 0` of `#[repr(C, align(64))]`
    /// [`CrystallizedAnchorRecord`], guaranteed 64-byte aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `CrystallizedAnchorRecord` is `#[repr(C, align(64))]` and `phase_deltas` is at offset 0
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &mut *(self
                .phase_deltas
                .as_mut_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// All-zero 128-byte empty record constant enabling 512-byte GF(2) $L^{32}$ zero-block skipping
    /// during [`AttractorGrowthSnapshot::compute_payload_crc32`].
    pub const EMPTY: Self = Self {
        phase_deltas: [0; MAX_ATTENTION_HEADS],
        id: 0,
        anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
        domain_id: DOMAIN_SYSTEM_BASELINE,
        half_life_steps: 0,
        weight: 0.0,
        pull_force: 0.0,
        stigmergic_density: 0.0,
        resultant_r_bar: 0.0,
        von_mises_kappa: 0.0,
        is_active: 0,
        is_crystallized: 0,
        reserved_flags: 0,
        last_active_step: 0,
        reserved_padding: [0; 16],
    };

    /// Packs a full [`AttractorAnchor`] into a compact 128-byte [`CrystallizedAnchorRecord`],
    /// computing its 64-head circular statistics $(\bar{R}_c, \kappa_c)$ in constant time.
    #[inline]
    #[must_use]
    pub fn from_anchor(anchor: &AttractorAnchor) -> Self {
        let r_bar = circular_r_bar_from_words_64(anchor.phase_words_ref());
        let kappa = branchless_von_mises_kappa(r_bar);
        Self {
            phase_deltas: anchor.phase_deltas,
            id: anchor.id,
            anchor_type: anchor.anchor_type,
            domain_id: anchor.domain_id,
            half_life_steps: anchor.half_life_steps,
            weight: anchor.weight,
            pull_force: anchor.pull_force,
            stigmergic_density: anchor.stigmergic_density,
            resultant_r_bar: r_bar,
            von_mises_kappa: kappa,
            is_active: anchor.is_active,
            is_crystallized: anchor.is_crystallized,
            reserved_flags: 0,
            last_active_step: anchor.last_active_step,
            reserved_padding: [0; 16],
        }
    }

    /// Expands this 128-byte record back into a full 640-byte runtime [`AttractorAnchor`].
    #[inline]
    #[must_use]
    pub const fn to_anchor(&self) -> AttractorAnchor {
        let mut vector_data = [0.0_f32; MAX_EMBEDDING_DIM];
        let mut i = 0usize;
        while i < 16 {
            vector_data[i] = (self.phase_deltas[i] as f32) * (1.0 / 128.0);
            i += 1;
        }
        AttractorAnchor {
            id: self.id,
            anchor_type: self.anchor_type,
            weight: self.weight,
            pull_force: self.pull_force,
            stigmergic_density: self.stigmergic_density,
            half_life_steps: branchless_select_u32(
                self.half_life_steps == 0,
                1000,
                self.half_life_steps,
            ),
            last_active_step: self.last_active_step,
            domain_id: self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1),
            vector_dim: 16,
            is_active: self.is_active,
            is_crystallized: self.is_crystallized,
            reserved_padding: [0; 24],
            phase_deltas: self.phase_deltas,
            vector_data,
        }
    }

    /// Writes this 128-byte record directly into `dst` in-place, touching only Cache Line 0
    /// (`offset 0..64`), Cache Line 1 (`phase_deltas`, `offset 64..128`), and the first 16
    /// floats (`64` bytes) of `vector_data` without constructing a 640-byte stack temporary or
    /// zeroing the remaining 448 bytes of unused `vector_data[16..128]`.
    #[inline(always)]
    pub fn write_to_anchor_in_place(
        &self,
        dst: &mut AttractorAnchor,
        domain_override: Option<u32>,
    ) {
        dst.id = self.id;
        dst.anchor_type = self.anchor_type;
        dst.weight = self.weight;
        dst.pull_force = self.pull_force;
        dst.stigmergic_density = self.stigmergic_density;
        dst.half_life_steps =
            branchless_select_u32(self.half_life_steps == 0, 1000, self.half_life_steps);
        dst.last_active_step = self.last_active_step;
        dst.domain_id =
            domain_override.unwrap_or(self.domain_id) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1);
        dst.vector_dim = 16;
        dst.is_active = self.is_active;
        dst.is_crystallized = self.is_crystallized;
        dst.phase_deltas = self.phase_deltas;
        let mut i = 0usize;
        while i < 16 {
            dst.vector_data[i] = (self.phase_deltas[i] as f32) * (1.0 / 128.0);
            i += 1;
        }
    }

    #[allow(clippy::wrong_self_convention)]
    #[inline(always)]
    #[must_use]
    const fn to_crystallized_attractor_record_unstamped(&self) -> CrystallizedAttractorRecord {
        let mut rec = CrystallizedAttractorRecord::new_empty(
            0,
            (self.domain_id as u8) & ((MAX_ATTRACTOR_DOMAINS as u8) - 1),
            0,
        );
        rec.anchor_id = self.id;
        rec.phase_profile = self.phase_deltas;
        rec.sample_count = branchless_select_u32(self.is_active != 0, 1, 0);
        rec.base_energy = branchless_relu_f32(self.weight);
        rec.resultant_r_bar = branchless_clamp_f32(self.resultant_r_bar, 0.0, 1.0);
        rec.von_mises_kappa = branchless_relu_f32(self.von_mises_kappa);
        rec.phase_gravity_mass = branchless_clamp_f32(
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
            MAX_ATTRACTOR_PULL_FORCE,
        );
        rec.provenance_bits = CAUSAL_AXIS_SUBJECTIVE_PERSONA
            | branchless_select_u32(
                self.is_crystallized != 0,
                CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                0,
            );
        rec.dominant_cause_code = ROOT_CAUSE_NOMINAL;
        rec
    }

    /// Converts this snapshot [`CrystallizedAnchorRecord`] into a 2-Origin Gateway / Online Toroidal
    /// $k$-Means [`CrystallizedAttractorRecord`] (`128B` `Pod`), computing `checksum_crc32` once.
    #[inline]
    #[must_use]
    pub const fn to_crystallized_attractor_record(&self) -> CrystallizedAttractorRecord {
        let mut rec = self.to_crystallized_attractor_record_unstamped();
        rec.checksum_crc32 = rec.compute_payload_crc32_with_words(self.phase_words_ref());
        rec
    }

    /// Constructs a [`CrystallizedAnchorRecord`] from a 2-Origin Gateway / Online Toroidal $k$-Means
    /// [`CrystallizedAttractorRecord`].
    #[inline]
    #[must_use]
    pub const fn from_crystallized_attractor_record(rec: &CrystallizedAttractorRecord) -> Self {
        let active = branchless_select_u8(rec.sample_count > 0, 1, 0);
        Self {
            phase_deltas: rec.phase_profile,
            id: rec.anchor_id,
            anchor_type: GCSO_ANCHOR_TYPE_CRYSTALLIZED,
            domain_id: (rec.dominant_slot as u32) & ((MAX_ATTRACTOR_DOMAINS as u32) - 1),
            half_life_steps: 1000,
            weight: branchless_max_f32(rec.base_energy, 1.0),
            pull_force: branchless_max_f32(rec.phase_gravity_mass, 0.25),
            stigmergic_density: branchless_clamp_f32(
                rec.phase_gravity_mass,
                0.25,
                MAX_STIGMERGIC_DENSITY,
            ),
            resultant_r_bar: rec.resultant_r_bar,
            von_mises_kappa: rec.von_mises_kappa,
            is_active: active,
            is_crystallized: active,
            reserved_flags: 0,
            last_active_step: 0,
            reserved_padding: [0; 16],
        }
    }
}

impl CausalProvenanceSignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        branchless_select_u32(
            self.is_active != 0,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA
                | branchless_select_u32(
                    self.is_crystallized != 0,
                    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                    0,
                ),
            0,
        )
    }
}

impl CognitiveGroundingSignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_deltas)
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.sample_mass()
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.id
    }
}

impl SubjectivePersonaSignal for CrystallizedAnchorRecord {}

impl ManifoldGatewaySignal for CrystallizedAnchorRecord {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; crate::traits::NUM_SWARM_MICRO_CELLS]) {
        self.to_crystallized_attractor_record_unstamped()
            .synthesize_gateway_frame_with_words()
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        self.to_crystallized_attractor_record_unstamped()
            .synthesize_gateway_frame()
    }
}

/// Self-contained 8,448-byte (`8.25 KB`, 132 cachelines) 64-byte-aligned [`Pod`] snapshot structure
/// for `.gcso` ZIMMS persistence and Steam Workshop distribution of a grown personality.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorGrowthSnapshot {
    /// Binary magic constant (`GCSO_SNAPSHOT_MAGIC_U32 = 0x4F534347`, offset 0..4).
    pub magic: u32,
    /// Unified ABI version (`GCSO_ABI_VERSION_HEX = 0x00000101`, offset 4..8).
    pub abi_version: u32,
    /// Number of active anchor records stored in `anchors` (`0..=64`, offset 8..12).
    pub active_anchor_count: u32,
    /// Number of bottom-up crystallized anchors in `anchors` (offset 12..16).
    pub crystallized_anchor_count: u32,
    /// Global token step counter at snapshot creation (offset 16..24).
    pub global_step_counter: u64,
    /// Active $8 \times 8$ cross-slot coupling bitmatrix (offset 24..32).
    pub coupling_matrix_8x8: u64,
    /// Synthesized persona resultant vector length $\bar{R}_c \in [0, 1]$ (offset 32..36).
    pub composite_r_bar: f32,
    /// Synthesized persona von Mises concentration $\kappa_c$ (offset 36..40).
    pub composite_kappa: f32,
    /// Total potential energy across the attractor space (offset 40..44).
    pub total_potential_energy: f32,
    /// IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]` (offset 44..48).
    pub payload_crc32: u32,
    /// Active working lane bitmask (`0..=0xFF`, offset 48..52).
    pub active_domains_mask: u32,
    /// Global anti-hallucination repulsion gain factor (offset 52..56).
    pub repulsion_gain: f32,
    /// Explicit header padding to 64-byte cacheline boundary (offset 56..64).
    pub reserved_header: [u8; 8],
    /// Synthesized persistent 64-head Q7 persona phase profile (offset 64..128).
    pub persona_phase_profile: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Global anti-phase repulsion profile vector (offset 128..192).
    pub repulsion_phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Relative working lane weights (`8 * 4 = 32` bytes, offset 192..224).
    pub domain_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Explicit metadata padding to 64-byte cacheline boundary (offset 224..256).
    pub reserved_meta: [u8; 32],
    /// Compact crystallized/active anchor records (`64 * 128 = 8192` bytes, offset 256..8448).
    pub anchors: [CrystallizedAnchorRecord; MAX_ANCHORS_PER_FIELD],
}

// SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` of size 8448 where all fields are `Pod`.
unsafe impl Pod for AttractorGrowthSnapshot {}

impl Default for AttractorGrowthSnapshot {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl AttractorGrowthSnapshot {
    /// Borrows `self.persona_phase_profile` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorGrowthSnapshot`], guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn persona_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` and `persona_phase_profile`
        // is at byte offset 64 (64-byte aligned, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .persona_phase_profile
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Borrows `self.repulsion_phase_deltas` (`offset 128` of `#[repr(C, align(64))]`
    /// [`AttractorGrowthSnapshot`], guaranteed 64-byte aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorGrowthSnapshot` is `#[repr(C, align(64))]` and `repulsion_phase_deltas`
        // is at byte offset 128 (64-byte aligned, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .repulsion_phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Constructs an empty, header-initialized `AttractorGrowthSnapshot` stamped with magic `"GCSO"`
    /// (`0x4F534347`) and ABI version `0.1.1` (`0x00000101`).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            magic: GCSO_SNAPSHOT_MAGIC_U32,
            abi_version: GCSO_ABI_VERSION_HEX,
            active_anchor_count: 0,
            crystallized_anchor_count: 0,
            global_step_counter: 0,
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            composite_r_bar: 0.0,
            composite_kappa: 0.0,
            total_potential_energy: 0.0,
            payload_crc32: 0,
            active_domains_mask: 0xFF,
            repulsion_gain: 0.0,
            reserved_header: [0; 8],
            persona_phase_profile: [0; MAX_ATTENTION_HEADS],
            repulsion_phase_deltas: [0; MAX_ATTENTION_HEADS],
            domain_weights: [1.0; MAX_ATTRACTOR_DOMAINS],
            reserved_meta: [0; 32],
            anchors: [CrystallizedAnchorRecord::EMPTY; MAX_ANCHORS_PER_FIELD],
        }
    }

    #[inline(always)]
    const fn is_zero_anchor_blocks(b0: &[u64; 8], b1: &[u64; 8]) -> bool {
        if (b1[0] | b1[1] | b1[3]) != 0 {
            return false;
        }
        (((b0[0] | b0[1]) | (b0[2] | b0[3]))
            | ((b0[4] | b0[5]) | (b0[6] | b0[7]))
            | ((b1[2] | b1[4]) | (b1[5] | b1[6]) | b1[7]))
            == 0
    }

    /// Computes the IEEE 802.3 CRC32 checksum over the 64-byte-aligned payload bytes `[64..8448]`
    /// (`131` 64-byte blocks) by aligning the 3 fixed metadata blocks (`blocks[0..3]`) first and
    /// stepping across 4-anchor (`512B`) and 2-anchor (`256B`) zero blocks via GF(2) $L^{32}$ / $L^{16}$ operators.
    #[inline]
    #[must_use]
    pub const fn compute_payload_crc32(&self) -> u32 {
        const PAYLOAD_BLOCKS_64: usize =
            (size_of::<AttractorGrowthSnapshot>() - SNAPSHOT_HEADER_BYTES) / 64;
        // SAFETY: `self` is `#[repr(C, align(64))]` of size 8448 bytes; offset 64 is 64-byte aligned
        // and followed by exactly `131` initialized 64-byte (`[u64; 8]`) blocks.
        let blocks: &[[u64; 8]; PAYLOAD_BLOCKS_64] = unsafe {
            &*(self as *const Self)
                .cast::<u8>()
                .add(SNAPSHOT_HEADER_BYTES)
                .cast::<[[u64; 8]; PAYLOAD_BLOCKS_64]>()
        };
        let c0 = crc32_step_block64_words(0xFFFF_FFFF_u32, &blocks[0]);
        let c1 = crc32_step_block64_words(c0, &blocks[1]);
        let mut crc = crc32_step_block64_words(c1, &blocks[2]);

        let mut i = 3usize;
        while i + 8 <= PAYLOAD_BLOCKS_64 {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                if Self::is_zero_anchor_blocks(&blocks[i + 4], &blocks[i + 5])
                    && Self::is_zero_anchor_blocks(&blocks[i + 6], &blocks[i + 7])
                {
                    crc = crc32_step_l32(crc);
                    i += 8;
                    continue;
                }
                crc = crc32_step_l16(crc);
                i += 4;
                continue;
            }
            let p01 = if z_a0 {
                0
            } else {
                let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                crc32_step_block64_words(z0, &blocks[i + 1])
            };
            let p23 = if z_a1 {
                0
            } else {
                let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                crc32_step_block64_words(a2, &blocks[i + 3])
            };
            crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            i += 4;
        }
        while i + 4 <= PAYLOAD_BLOCKS_64 {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                crc = crc32_step_l16(crc);
            } else {
                let p01 = if z_a0 {
                    0
                } else {
                    let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                    crc32_step_block64_words(z0, &blocks[i + 1])
                };
                let p23 = if z_a1 {
                    0
                } else {
                    let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                    crc32_step_block64_words(a2, &blocks[i + 3])
                };
                crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            }
            i += 4;
        }
        !crc
    }

    /// Computes the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]` when
    /// `self.anchors[active_anchors..64]` is guaranteed to be 100% zero bytes
    /// ([`CrystallizedAnchorRecord::EMPTY`]), folding the trailing `(64 - active_anchors)`
    /// zero anchor slots via $\mathcal{O}(1)$ GF(2) $L^{32}$ / $L^{16}$ / $L^8$ operators
    /// without reading `7.8 KB` of zero-initialized tail memory.
    #[inline]
    #[must_use]
    pub const fn compute_payload_crc32_with_known_zero_tail(&self, active_anchors: usize) -> u32 {
        const PAYLOAD_BLOCKS_64: usize =
            (size_of::<AttractorGrowthSnapshot>() - SNAPSHOT_HEADER_BYTES) / 64;
        let active = if active_anchors < MAX_ANCHORS_PER_FIELD {
            active_anchors
        } else {
            MAX_ANCHORS_PER_FIELD
        };
        // SAFETY: `self` is `#[repr(C, align(64))]` of size 8448 bytes; offset 64 is 64-byte aligned
        // and followed by `131` initialized 64-byte (`[u64; 8]`) blocks.
        let blocks: &[[u64; 8]; PAYLOAD_BLOCKS_64] = unsafe {
            &*(self as *const Self)
                .cast::<u8>()
                .add(SNAPSHOT_HEADER_BYTES)
                .cast::<[[u64; 8]; PAYLOAD_BLOCKS_64]>()
        };
        let c0 = crc32_step_block64_words(0xFFFF_FFFF_u32, &blocks[0]);
        let c1 = crc32_step_block64_words(c0, &blocks[1]);
        let mut crc = crc32_step_block64_words(c1, &blocks[2]);

        let active_end_block = 3 + active * 2;
        let mut i = 3usize;
        while i + 4 <= active_end_block {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            let z_a1 = Self::is_zero_anchor_blocks(&blocks[i + 2], &blocks[i + 3]);
            if z_a0 && z_a1 {
                crc = crc32_step_l16(crc);
            } else {
                let p01 = if z_a0 {
                    0
                } else {
                    let z0 = crc32_block64_zero_seed_words(&blocks[i]);
                    crc32_step_block64_words(z0, &blocks[i + 1])
                };
                let p23 = if z_a1 {
                    0
                } else {
                    let a2 = crc32_block64_zero_seed_words(&blocks[i + 2]);
                    crc32_step_block64_words(a2, &blocks[i + 3])
                };
                crc = crc32_step_l16(crc) ^ (if p01 == 0 { 0 } else { crc32_step_l8(p01) }) ^ p23;
            }
            i += 4;
        }
        if i + 2 <= active_end_block {
            let z_a0 = Self::is_zero_anchor_blocks(&blocks[i], &blocks[i + 1]);
            if z_a0 {
                crc = crc32_step_l8(crc);
            } else {
                let z0 = crc32_step_block64_words(crc, &blocks[i]);
                crc = crc32_step_block64_words(z0, &blocks[i + 1]);
            }
        }
        let mut rem_zero_anchors = MAX_ANCHORS_PER_FIELD - active;
        while rem_zero_anchors >= 4 {
            crc = crc32_step_l32(crc);
            rem_zero_anchors -= 4;
        }
        if rem_zero_anchors >= 2 {
            crc = crc32_step_l16(crc);
            rem_zero_anchors -= 2;
        }
        if rem_zero_anchors >= 1 {
            crc = crc32_step_l8(crc);
        }
        !crc
    }

    /// Computes and stamps the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]`.
    #[inline]
    pub fn finalize_crc32(&mut self) -> u32 {
        let crc = self.compute_payload_crc32();
        self.payload_crc32 = crc;
        crc
    }

    /// Computes and stamps the IEEE 802.3 CRC32 checksum over payload bytes `[64..8448]`
    /// when `self.anchors[active_anchors..64]` is guaranteed to be zeroed.
    #[inline]
    pub fn finalize_crc32_for_active_anchors(&mut self, active_anchors: usize) -> u32 {
        let crc = self.compute_payload_crc32_with_known_zero_tail(active_anchors);
        self.payload_crc32 = crc;
        crc
    }

    /// Verifies the snapshot magic (`0x4F534347`), ABI version (`0x00000101`), anchor count bounds,
    /// and IEEE 802.3 CRC32 payload checksum.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] if magic, bounds, or CRC32 verification fails.
    /// Returns [`GCSO_ERROR_VERSION_MISMATCH`] if `abi_version != GCSO_ABI_VERSION_HEX`.
    #[inline]
    pub fn verify_integrity(&self) -> GcsoResult<()> {
        if self.magic != GCSO_SNAPSHOT_MAGIC_U32 {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        if self.abi_version != GCSO_ABI_VERSION_HEX {
            return Err(GCSO_ERROR_VERSION_MISMATCH);
        }
        if (self.active_anchor_count as usize) > MAX_ANCHORS_PER_FIELD
            || self.crystallized_anchor_count > self.active_anchor_count
        {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        let computed_crc = self.compute_payload_crc32();
        if self.payload_crc32 != computed_crc {
            return Err(GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        Ok(())
    }

    /// Zero-copy casts and verifies an `AttractorGrowthSnapshot` directly from a 64-byte-aligned
    /// `.gcso` ZIMMS memory-mapped slice at `offset`.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `bytes` cannot hold `size_of::<Self>()`,
    /// [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`] if `offset` is not 64-byte aligned, or
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`] / [`GCSO_ERROR_VERSION_MISMATCH`] if integrity checks fail.
    #[inline]
    pub fn from_bytes_verified_at(bytes: &[u8], offset: usize) -> GcsoResult<&Self> {
        let snap = Self::from_bytes_at(bytes, offset)?;
        snap.verify_integrity()?;
        Ok(snap)
    }
}

// ===================================================================
// 4. Domain Accumulator Summary & Single Phase-Field Anchor
// ===================================================================

/// Summary descriptor capturing accumulator pressure metrics for a single attractor lane.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DomainAccumulatorSummary {
    /// Lane / domain identifier (`0..=7`).
    pub domain_id: u32,
    /// Number of active anchors registered in this lane.
    pub active_count: u32,
    /// Maximum anchor capacity of this lane (`<= 64`).
    pub capacity: u32,
    /// Number of bottom-up crystallized (invariant) anchors.
    pub crystallized_count: u32,
    /// Accumulator pressure ratio (`active_count / capacity`).
    pub pressure_ratio: f32,
    /// Total potential energy accumulated across active anchors.
    pub total_potential_energy: f32,
    /// Accumulated energy dissipated during evaporative displacement.
    pub dissipated_energy_total: f32,
    /// Autonomous or user-configured relative superposition weight scalar.
    pub domain_weight: f32,
}

// SAFETY: `DomainAccumulatorSummary` is a 32-byte `#[repr(C, align(16))]` struct of primitive `u32`/`f32` fields.
unsafe impl Pod for DomainAccumulatorSummary {}

/// Single topological anchor point living on the unified $SO(2)^{64}$ phase field (`640` bytes = `10` cache lines).
///
/// # 3-Tier Cache-Line Layout (`#[repr(C, align(64))]`)
/// - **Cache Line 0 (`offset 0..64`)**: Hot scalar metadata (`id`, `anchor_type`, `weight`,
///   `pull_force`, `stigmergic_density`, `half_life_steps`, `last_active_step`, `domain_id`,
///   `vector_dim`, `is_active`, `is_crystallized`, `reserved_padding`) so active-slot screening
///   and `compute_energy_score` touch strictly a single 64-byte L1D cache line.
/// - **Cache Line 1 (`offset 64..128`)**: `phase_deltas: [gcso_q7_t; 64]`, strictly 64-byte
///   cacheline-aligned so 8-word SWAR (`phase_words_ref`) and AVX2 loads never cross a cache line.
/// - **Cache Lines 2..9 (`offset 128..640`)**: Cold `vector_data: [f32; 128]` (`512` bytes),
///   untouched during hot/mezzo phase folding.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorAnchor {
    /// Unique anchor identifier (1-based index, `0` indicates unallocated slot).
    pub id: u32,
    /// Classification type of topological anchor (`gcso_anchor_type_t`).
    pub anchor_type: gcso_anchor_type_t,
    /// Intrinsic mass / baseline salience of the anchor.
    pub weight: f32,
    /// Dynamic gravitational pull force exerted by the anchor ($F_{\mathrm{pull}}$).
    pub pull_force: f32,
    /// Accumulated stigmergic trail density ($\rho_{\mathrm{stigmergy}}$).
    pub stigmergic_density: f32,
    /// Half-life in token steps for temporal energy decay under accumulator pressure.
    pub half_life_steps: u32,
    /// Token step index when this anchor was last activated or traversed.
    pub last_active_step: u64,
    /// Originating working lane identifier (`0..=7`) associated with this anchor.
    pub domain_id: u32,
    /// Active dimension of stored vector data (`0..=128`).
    pub vector_dim: u16,
    /// Flag indicating whether anchor is active (`1` = active, `0` = inactive).
    pub is_active: u8,
    /// Flag indicating whether anchor was bottom-up crystallized from high-density trails (`1` or `0`).
    pub is_crystallized: u8,
    /// Explicit padding completing Cache Line 0 (`offset 40..64`).
    pub reserved_padding: [u8; 24],
    /// Target phase coordinates across 64 attention heads (Q7 fixed-point, Cache Line 1: `offset 64..128`).
    pub phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Continuous feature embedding vector buffer (up to 128 dimensions, Cache Lines 2..9: `offset 128..640`).
    pub vector_data: [f32; MAX_EMBEDDING_DIM],
}

// SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` with explicit padding and primitive fields where all bit patterns are valid.
unsafe impl Pod for AttractorAnchor {}

impl AttractorAnchor {
    /// Deactivates this anchor slot in-place by clearing only its active/crystallized flags,
    /// ID, and scalar weights (`20` bytes instead of writing a full `640`-byte `AttractorAnchor::default()`),
    /// eliminating `620` bytes of dead `vector_data` / `phase_deltas` memory traffic per deactivated slot.
    #[inline(always)]
    pub const fn deactivate_in_place(&mut self) {
        self.id = 0;
        self.weight = 0.0;
        self.pull_force = 0.0;
        self.stigmergic_density = 0.0;
        self.is_active = 0;
        self.is_crystallized = 0;
    }
}

impl Default for AttractorAnchor {
    #[inline(always)]
    fn default() -> Self {
        Self {
            id: 0,
            anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
            weight: 1.0,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            half_life_steps: 1000,
            last_active_step: 0,
            domain_id: DOMAIN_SYSTEM_BASELINE,
            vector_dim: 0,
            is_active: 0,
            is_crystallized: 0,
            reserved_padding: [0; 24],
            phase_deltas: [0; MAX_ATTENTION_HEADS],
            vector_data: [0.0; MAX_EMBEDDING_DIM],
        }
    }
}

impl EpisodicPhaseSample for AttractorAnchor {
    #[inline(always)]
    fn sample_phase_coords(&self, _domain_salt: u32) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn sample_mass(&self) -> f32 {
        branchless_select_f32(
            self.is_active != 0,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force),
            0.0,
        )
    }

    #[inline(always)]
    fn sample_coherence_prior(&self) -> f32 {
        let r_bar = circular_r_bar_from_words_64(self.phase_words_ref());
        branchless_clamp_f32(r_bar, 0.25, 1.0)
    }

    #[inline(always)]
    fn preferred_working_lane(&self) -> u32 {
        self.domain_id & ((MAX_ATTRACTOR_DOMAINS as u32) - 1)
    }

    #[inline(always)]
    fn sample_anchor_type(&self) -> gcso_anchor_type_t {
        self.anchor_type
    }
}

impl CausalProvenanceSignal for AttractorAnchor {
    #[inline(always)]
    fn causal_provenance_bits(&self) -> u32 {
        branchless_select_u32(
            self.is_active != 0,
            CAUSAL_AXIS_SUBJECTIVE_PERSONA
                | branchless_select_u32(
                    self.is_crystallized != 0,
                    CAUSAL_CAUSE_ATTRACTOR_CRYSTALLIZED,
                    0,
                ),
            0,
        )
    }
}

impl CognitiveGroundingSignal for AttractorAnchor {
    #[inline(always)]
    fn project_grounding_phases_64(&self) -> [gcso_q7_t; MAX_ATTENTION_HEADS] {
        self.phase_deltas
    }

    #[inline(always)]
    fn grounding_phases_ref_64(&self) -> Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]> {
        Some(&self.phase_deltas)
    }

    #[inline(always)]
    fn project_grounding_words_64(&self) -> [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        *self.phase_words_ref()
    }

    #[inline(always)]
    fn grounding_base_energy(&self) -> f32 {
        self.sample_mass()
    }

    #[inline(always)]
    fn grounding_anchor_id(&self) -> u32 {
        self.id
    }
}

impl SubjectivePersonaSignal for AttractorAnchor {}

impl ManifoldGatewaySignal for AttractorAnchor {
    #[inline(always)]
    fn synthesize_gateway_frame_with_words(
        &self,
    ) -> (GatewayFrame, [u64; crate::traits::NUM_SWARM_MICRO_CELLS]) {
        if self.is_active == 0 {
            return (
                GatewayFrame::zeroed(),
                [0u64; crate::traits::NUM_SWARM_MICRO_CELLS],
            );
        }
        CrystallizedAnchorRecord::from_anchor(self).synthesize_gateway_frame_with_words()
    }

    #[inline(always)]
    fn synthesize_gateway_frame(&self) -> GatewayFrame {
        if self.is_active == 0 {
            return GatewayFrame::zeroed();
        }
        CrystallizedAnchorRecord::from_anchor(self).synthesize_gateway_frame()
    }
}

impl AttractorAnchor {
    /// Zero-copy `const` reference reinterpretation of `self.phase_deltas` (`offset 64..128`,
    /// guaranteed 64-byte cacheline-aligned by `#[repr(C, align(64))]`) as 8 native `u64` SWAR words (`&[u64; 8]`).
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &*(self
                .phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Mutably borrows `self.phase_deltas` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorAnchor`], guaranteed 64-byte cacheline-aligned) as a zero-copy `&mut [u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn phase_words_mut(&mut self) -> &mut [u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorAnchor` is `#[repr(C, align(64))]` and `phase_deltas` is at byte offset 64
        // (64-byte aligned, >= 8-byte alignment of `u64`), with identical size (64 bytes) and `Pod` validity.
        unsafe {
            &mut *(self
                .phase_deltas
                .as_mut_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Computes composite potential energy score with division-free (`0 fdiv`) branchless
    /// temporal decay under Accumulator Pressure.
    #[inline(always)]
    #[must_use]
    pub const fn compute_energy_score(&self, current_step: u64) -> f32 {
        let active = self.is_active != 0;
        let dt = current_step.saturating_sub(self.last_active_step) as f32;
        let half_life = branchless_select_f32(
            self.half_life_steps == 0,
            1000.0,
            self.half_life_steps as f32,
        );
        let inv_half_life = fast_inv_pos_normal_f32(half_life);
        let temporal_decay = fast_inv_pos_normal_f32(1.0 + dt * inv_half_life);
        let clean_w = branchless_relu_f32(self.weight);
        let clean_p = branchless_relu_f32(self.pull_force);
        let clean_rho = branchless_clamp_f32(self.stigmergic_density, 0.0, MAX_STIGMERGIC_DENSITY);
        let base_energy = clean_w * clean_p + clean_rho;
        let energy = branchless_flush_subnormal_f32(base_energy * temporal_decay);
        branchless_select_f32(active, energy, 0.0)
    }

    /// Evaluates **Zero-Hyperparameter Phase-Proximity Gravity** on the unified phase field
    /// in strictly constant time (`0` branches on phase non-zero checks, `0 fdiv`) by delegating
    /// to register-resident [`branchless_q7_resonant_pull_64`].
    ///
    /// Returns `(effective_pull_force, phase_coherence)`.
    #[inline(always)]
    #[must_use]
    pub fn compute_resonant_pull(
        &self,
        current_step: u64,
        trail: &gcso_pointer_trail_t,
    ) -> (f32, f32) {
        let active = self.is_active != 0;
        let energy = self.compute_energy_score(current_step);
        let base_pull = branchless_select_f32(
            active,
            branchless_relu_f32(self.weight) * branchless_relu_f32(self.pull_force) * energy,
            0.0,
        );
        let (res_pull, coh) = branchless_q7_resonant_pull_64(
            &trail.accumulated_phase_delta,
            &self.phase_deltas,
            base_pull,
        );
        (
            branchless_select_f32(
                active,
                branchless_min_f32(res_pull, MAX_ATTRACTOR_PULL_FORCE),
                0.0,
            ),
            branchless_select_f32(active, coh, 0.0),
        )
    }

    /// Recharges anchor activation timestamp and boosts stigmergic density bounded by [`MAX_STIGMERGIC_DENSITY`].
    #[inline(always)]
    pub fn touch(&mut self, current_step: u64, density_delta: f32) {
        self.last_active_step = current_step;
        let boost = branchless_flush_subnormal_f32(branchless_relu_f32(density_delta));
        self.stigmergic_density = branchless_min_f32(
            branchless_relu_f32(self.stigmergic_density) + boost,
            MAX_STIGMERGIC_DENSITY,
        );
    }
}

// ===================================================================
// 5. Single-Lane Attractor Field Accumulator Engine
// ===================================================================

/// Dynamic Attractor Field Engine with Accumulator Pressure Dynamics and $S^1$ Geodesic Phase-Proximity Gravity.
///
/// # Cache-Line Layout (`#[repr(C, align(64))]`)
/// - **Cache Line 0 (`offset 0..64`)**: Hot lane control metadata (`active_count`, `capacity`,
///   `domain_id`, `repulsion_gain`, `accumulator_pressure`, `evaporation_rate`,
///   `dissipated_energy_total`, `next_anchor_id`, `step_counter`, `reserved_padding`).
/// - **Cache Line 1 (`offset 64..128`)**: `repulsion_phase_deltas: [gcso_q7_t; 64]`, strictly
///   64-byte cacheline-aligned (`0` cacheline splits).
/// - **Cache Lines 2..641 (`offset 128..41088`)**: `anchors: [AttractorAnchor; 64]`, each strictly
///   64-byte cacheline-aligned (`640` bytes).
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorFieldEngine {
    /// Number of active anchors registered in this lane.
    pub active_count: u32,
    /// Maximum capacity of this lane (`<= 64`).
    pub capacity: u32,
    /// Target working lane identifier (`0..=7`).
    pub domain_id: u32,
    /// Global phase repulsion gain factor.
    pub repulsion_gain: f32,
    /// Accumulator pressure factor controlling minimum threshold for evaporative displacement.
    pub accumulator_pressure: f32,
    /// Evaporation rate per step applied to stigmergic density under external pressure.
    pub evaporation_rate: f32,
    /// Cumulative energy dissipated through anchor evaporation or displacement.
    pub dissipated_energy_total: f32,
    /// Auto-incrementing anchor ID generator.
    next_anchor_id: u32,
    /// Internal token step counter for LRU and temporal decay tracking.
    pub step_counter: u64,
    /// Explicit padding completing Cache Line 0 (`offset 40..64`).
    pub reserved_padding: [u8; 24],
    /// Global anti-phase repulsion profile vector ($-\boldsymbol{\Delta\theta}$, Cache Line 1: `offset 64..128`).
    pub repulsion_phase_deltas: [gcso_q7_t; MAX_ATTENTION_HEADS],
    /// Fixed-capacity storage array for topological anchors (`64` slots, `offset 128..41088`).
    pub anchors: [AttractorAnchor; MAX_ANCHORS_PER_FIELD],
}

// SAFETY: `AttractorFieldEngine` is `#[repr(C, align(64))]` where all fields are `Pod`.
unsafe impl Pod for AttractorFieldEngine {}

impl Default for AttractorFieldEngine {
    #[inline(always)]
    fn default() -> Self {
        Self::new(DOMAIN_SYSTEM_BASELINE, MAX_ANCHORS_PER_FIELD as u32)
    }
}

impl AttractorFieldEngine {
    /// Borrows `self.repulsion_phase_deltas` (`offset 64` of `#[repr(C, align(64))]`
    /// [`AttractorFieldEngine`], guaranteed 64-byte cacheline-aligned) as a zero-copy `&[u64; 8]` SWAR array.
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_words_ref(&self) -> &[u64; crate::traits::NUM_SWARM_MICRO_CELLS] {
        // SAFETY: `AttractorFieldEngine` is `#[repr(C, align(64))]` and `repulsion_phase_deltas`
        // is at byte offset 64 (`64 % 64 == 0`, >= 8-byte alignment of `u64`).
        unsafe {
            &*(self
                .repulsion_phase_deltas
                .as_ptr()
                .cast::<[u64; crate::traits::NUM_SWARM_MICRO_CELLS]>())
        }
    }

    /// Creates a new `AttractorFieldEngine` instance for a specific working lane.
    #[inline]
    #[must_use]
    pub const fn new(domain_id: u32, capacity: u32) -> Self {
        let cap = branchless_select_u32(
            capacity > (MAX_ANCHORS_PER_FIELD as u32),
            MAX_ANCHORS_PER_FIELD as u32,
            capacity,
        );
        Self {
            active_count: 0,
            capacity: cap,
            domain_id,
            repulsion_gain: 0.0,
            accumulator_pressure: 0.5,
            evaporation_rate: 0.01,
            dissipated_energy_total: 0.0,
            next_anchor_id: 1,
            step_counter: 0,
            reserved_padding: [0; 24],
            repulsion_phase_deltas: [0; MAX_ATTENTION_HEADS],
            anchors: [AttractorAnchor {
                id: 0,
                anchor_type: GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
                weight: 1.0,
                pull_force: 1.0,
                stigmergic_density: 1.0,
                half_life_steps: 1000,
                last_active_step: 0,
                domain_id,
                vector_dim: 0,
                is_active: 0,
                is_crystallized: 0,
                reserved_padding: [0; 24],
                phase_deltas: [0; MAX_ATTENTION_HEADS],
                vector_data: [0.0; MAX_EMBEDDING_DIM],
            }; MAX_ANCHORS_PER_FIELD],
        }
    }

    /// Returns the active anchor count.
    #[inline(always)]
    #[must_use]
    pub const fn active_count(&self) -> u32 {
        self.active_count
    }

    /// Returns the working lane / domain identifier.
    #[inline(always)]
    #[must_use]
    pub const fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Returns `true` if this lane is saturated (`active_count >= capacity`).
    #[inline(always)]
    #[must_use]
    pub const fn is_saturated(&self) -> bool {
        self.active_count >= self.capacity
    }

    /// Returns the number of crystallized anchors in this lane.
    #[inline]
    #[must_use]
    pub fn crystallized_count(&self) -> u32 {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return 0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut count = 0u32;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active != 0 {
                count = count.wrapping_add(u32::from(anchor.is_crystallized != 0));
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        count
    }

    /// Calculates the total potential energy currently stored in this lane.
    #[inline]
    #[must_use]
    pub fn total_potential_energy(&self) -> f32 {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return 0.0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut total = 0.0_f32;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active != 0 {
                total += anchor.compute_energy_score(self.step_counter);
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        branchless_flush_subnormal_f32(total)
    }

    /// Returns the lowest active non-crystallized anchor energy in this lane (or `0.0` if unsaturated).
    #[inline]
    #[must_use]
    pub fn min_displaceable_energy(&self) -> f32 {
        if !self.is_saturated() {
            return 0.0;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut lowest = f32::MAX;
        for anchor in &self.anchors[..cap] {
            if anchor.is_active == 0 {
                return 0.0;
            }
            if anchor.is_crystallized == 0 {
                let e = anchor.compute_energy_score(self.step_counter);
                if e < lowest {
                    lowest = e;
                }
            }
        }
        lowest
    }

    /// Calculates the current pressure ratio (`active_count / capacity`) in division-free (`0 fdiv`) mode.
    #[inline(always)]
    #[must_use]
    pub const fn pressure_ratio(&self) -> f32 {
        let cap = self.capacity as usize;
        branchless_flush_subnormal_f32((self.active_count as f32) * fast_inv_len_f32(cap))
    }

    /// Exports accumulator pressure metrics summary for this lane.
    #[inline]
    #[must_use]
    pub fn accumulator_summary(&self, domain_weight: f32) -> DomainAccumulatorSummary {
        DomainAccumulatorSummary {
            domain_id: self.domain_id,
            active_count: self.active_count,
            capacity: self.capacity,
            crystallized_count: self.crystallized_count(),
            pressure_ratio: self.pressure_ratio(),
            total_potential_energy: self.total_potential_energy(),
            dissipated_energy_total: self.dissipated_energy_total,
            domain_weight,
        }
    }

    /// Evaluates evaporative pressure decay across all active non-crystallized anchors in-place.
    #[inline]
    pub fn apply_evaporative_decay(&mut self) {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let retain = branchless_clamp_f32(1.0 - self.evaporation_rate, 0.0, 1.0);
        let min_energy = 0.05 * branchless_relu_f32(self.accumulator_pressure);
        let mut deactivated_count = 0u32;

        for anchor in &mut self.anchors[..cap] {
            if anchor.is_active == 0 {
                continue;
            }
            if anchor.is_crystallized == 0 {
                anchor.stigmergic_density =
                    branchless_flush_subnormal_f32(anchor.stigmergic_density * retain);
                let current_energy = anchor.compute_energy_score(self.step_counter);
                if current_energy < min_energy {
                    self.dissipated_energy_total += current_energy;
                    anchor.deactivate_in_place();
                    deactivated_count = deactivated_count.wrapping_add(1);
                }
            }
            remaining -= 1;
            if remaining == 0 {
                break;
            }
        }
        self.active_count = self.active_count.saturating_sub(deactivated_count);
    }

    /// Finds the weakest non-crystallized anchor slot (or oldest LRU slot if all are crystallized).
    #[inline]
    fn displace_weakest_anchor_slot(&self) -> Option<usize> {
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut lowest_score = f32::MAX;
        let mut target_idx = None;
        let mut oldest_step = u64::MAX;
        let mut lru_idx = None;

        for (idx, anchor) in self.anchors[..cap].iter().enumerate() {
            if anchor.is_active == 0 {
                return Some(idx);
            }
            if anchor.is_crystallized == 0 {
                let score = anchor.compute_energy_score(self.step_counter);
                if score < lowest_score {
                    lowest_score = score;
                    target_idx = Some(idx);
                }
            }
            if anchor.last_active_step < oldest_step {
                oldest_step = anchor.last_active_step;
                lru_idx = Some(idx);
            }
        }
        target_idx.or(lru_idx)
    }

    /// Allocates an anchor slot enforcing Accumulator Pressure Dynamics (Evaporative Displacement).
    #[inline]
    fn allocate_anchor_slot(&mut self, incoming_energy: f32) -> GcsoResult<usize> {
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        if !self.is_saturated() {
            for (idx, anchor) in self.anchors[..cap].iter().enumerate() {
                if anchor.is_active == 0 {
                    self.active_count = self.active_count.wrapping_add(1);
                    return Ok(idx);
                }
            }
        }

        let slot_idx = self
            .displace_weakest_anchor_slot()
            .ok_or(GCSO_ERROR_ACTION_HUB_FULL)?;
        let target_anchor = &self.anchors[slot_idx];
        if target_anchor.is_active != 0 {
            let existing_score = target_anchor.compute_energy_score(self.step_counter);
            if incoming_energy < existing_score * self.accumulator_pressure {
                return Err(GCSO_ERROR_ACTION_HUB_FULL);
            }
            self.dissipated_energy_total += existing_score;
        }
        Ok(slot_idx)
    }

    /// Internal helper registering an anchor with optional embedding or deterministic prompt hash phases.
    #[inline]
    fn insert_anchor_internal(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
        embedding: Option<&AlignedSlice32<'_, f32>>,
        prompt_phases: Option<&[gcso_q7_t; MAX_ATTENTION_HEADS]>,
        is_crystallized: bool,
    ) -> GcsoResult<u32> {
        if !branchless_is_finite_f32(weight) || weight <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }

        let slot_idx = self.allocate_anchor_slot(weight)?;
        let anchor_id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.wrapping_add(1);
        if self.next_anchor_id == 0 {
            self.next_anchor_id = 1;
        }

        let step = self.step_counter;
        let dom_id = self.domain_id;
        let dst = &mut self.anchors[slot_idx];
        dst.id = anchor_id;
        dst.anchor_type = anchor_type;
        dst.weight = weight;
        dst.pull_force = 1.0;
        dst.stigmergic_density = 1.0;
        dst.half_life_steps = 1000;
        dst.last_active_step = step;
        dst.domain_id = dom_id;
        dst.vector_dim = 0;
        dst.is_active = 1;
        dst.is_crystallized = u8::from(is_crystallized);

        let mut phases = match prompt_phases {
            Some(p_phases) => *p_phases,
            None => [0i8; MAX_ATTENTION_HEADS],
        };
        if let Some(emb) = embedding {
            let copy_len = emb.len().min(MAX_EMBEDDING_DIM);
            if copy_len > 0 {
                let src_slice = &emb.as_slice()[..copy_len];
                dst.vector_data[..copy_len].copy_from_slice(src_slice);
                dst.vector_dim = copy_len as u16;
                if prompt_phases.is_none() {
                    phases = project_f32_slice_to_q7_phases(src_slice);
                }
            }
        }
        if anchor_type == GCSO_ANCHOR_TYPE_PHASE_REPULSE {
            phases = branchless_q7_conjugate_64(&phases);
        }
        dst.phase_deltas = phases;
        Ok(anchor_id)
    }

    /// Directly registers any [`EpisodicPhaseSample`] as an anchor in this lane.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if sample mass is non-positive, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if displacement fails under accumulator pressure.
    #[inline]
    pub fn add_episodic_sample_anchor<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        is_crystallized: bool,
    ) -> GcsoResult<u32> {
        let phases = sample.sample_phase_coords(self.domain_id);
        let mass = sample.sample_mass();
        self.insert_anchor_internal(
            sample.sample_anchor_type(),
            mass,
            None,
            Some(&phases),
            is_crystallized,
        )
    }

    /// Finds an immutable anchor by unique anchor ID in this lane.
    #[inline]
    #[must_use]
    pub fn get_anchor(&self, anchor_id: u32) -> Option<&AttractorAnchor> {
        if anchor_id == 0 {
            return None;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        self.anchors[..cap]
            .iter()
            .find(|a| (a.is_active != 0) & (a.id == anchor_id))
    }

    /// Finds a mutable anchor by unique anchor ID in this lane.
    #[inline]
    pub fn get_anchor_mut(&mut self, anchor_id: u32) -> Option<&mut AttractorAnchor> {
        if anchor_id == 0 {
            return None;
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        self.anchors[..cap]
            .iter_mut()
            .find(|a| (a.is_active != 0) & (a.id == anchor_id))
    }

    /// Deactivates and removes an anchor by anchor ID.
    #[inline]
    pub fn remove_anchor(&mut self, anchor_id: u32) -> bool {
        if anchor_id == 0 {
            return false;
        }
        let step = self.step_counter;
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        if let Some(slot_idx) = self.anchors[..cap]
            .iter()
            .position(|a| (a.is_active != 0) & (a.id == anchor_id))
        {
            let score = self.anchors[slot_idx].compute_energy_score(step);
            self.dissipated_energy_total += score;
            self.anchors[slot_idx].deactivate_in_place();
            self.active_count = self.active_count.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Register-resident core evaluator returning `(composite_words_64, pull_force)` directly in
    /// 8 `u64` SWAR registers without intermediate memory stores or reloads.
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_composite_attractor_force_words(
        &self,
        init_w: &[u64; 8],
        init_nonzero: bool,
    ) -> ([u64; 8], f32) {
        if self.active_count == 0 && self.repulsion_gain <= 0.0 {
            return (*init_w, 0.0);
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let init_is_uniform = if init_nonzero {
            is_word_array_uniform_64(init_w)
        } else {
            true
        };
        let init_w0 = init_w[0];
        let init_b0 = (init_w0 & 0xFF) as u8;
        let init_is_byte_uniform = init_is_uniform & (init_w0 == init_w0.rotate_right(8));
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
        let mut run_w = *init_w;
        let mut run_is_uniform = init_is_uniform;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut run_w0 = init_w0;
        let mut running_pull = 0.0_f32;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut vi_vecs: Option<(
            core::arch::x86_64::__m256i,
            core::arch::x86_64::__m256i,
        )> = if init_is_uniform {
            None
        } else {
            // SAFETY: Guarded by `target_feature = "avx2"`.
            Some(unsafe {
                crate::traits::branchless::avx2_words64_to_vec64_with_uniform(init_w, false)
            })
        };
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut vr_vecs: Option<(
            core::arch::x86_64::__m256i,
            core::arch::x86_64::__m256i,
        )> = vi_vecs;

        let mut remaining = self.active_count;
        let mut idx = 0usize;
        while idx < cap && remaining > 0 {
            let anchor = &self.anchors[idx];
            if anchor.is_active != 0 {
                remaining -= 1;
                let energy = anchor.compute_energy_score(self.step_counter);
                let base_pull = branchless_relu_f32(anchor.weight)
                    * branchless_relu_f32(anchor.pull_force)
                    * energy;
                if base_pull > 1.0e-12 {
                    let anc_w = anchor.phase_words_ref();
                    let check_anc_uni = (init_nonzero & init_is_uniform)
                        | (running_pull <= 1.0e-12)
                        | run_is_uniform;
                    let anc_is_uniform = if check_anc_uni {
                        is_word_array_uniform_64(anc_w)
                    } else {
                        false
                    };
                    let anc_nonzero = if anc_is_uniform {
                        anc_w[0] != 0
                    } else {
                        words_any_nonzero_64(anc_w)
                    };
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    if anc_is_uniform & run_is_uniform & (init_is_uniform | !init_nonzero) {
                        let aw0 = anc_w[0];
                        let eff_coh = if init_nonzero & (aw0 != 0) {
                            if aw0 == init_w0 {
                                1.0
                            } else if init_is_byte_uniform && (aw0 == aw0.rotate_right(8)) {
                                Q7_FULL_COS_REF[init_b0.wrapping_sub(aw0 as u8) as usize]
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    init_w, true, anc_w, true,
                                )
                            }
                        } else {
                            0.0_f32
                        };
                        let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
                        let pull = branchless_min_f32(
                            branchless_flush_subnormal_f32(base_pull * proximity_gain),
                            MAX_ATTRACTOR_PULL_FORCE,
                        );
                        if pull > 1.0e-12 {
                            let total_w = running_pull + pull;
                            if running_pull <= 1.0e-12 {
                                run_w0 = aw0;
                            } else if run_w0 != aw0 {
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    run_w0 = aw0;
                                } else if s > 0 {
                                    let bias_restore = (128u64.wrapping_sub(s))
                                        .wrapping_mul(0x0101_0101_0101_0101);
                                    run_w0 =
                                        crate::traits::branchless::swar_slerp_word_unchecked_x8(
                                            run_w0,
                                            aw0,
                                            s,
                                            bias_restore,
                                        );
                                }
                            }
                            running_pull = total_w;
                        }
                        idx += 1;
                        continue;
                    }
                    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                    let (va0, va1) = unsafe {
                        crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                            anc_w,
                            anc_is_uniform,
                        )
                    };
                    let eff_coh = if init_nonzero & anc_nonzero {
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            unsafe {
                                use core::arch::x86_64::_mm256_sub_epi8;
                                let (vi0, vi1) = match vi_vecs {
                                    Some(v) => v,
                                    None => {
                                        let v = crate::traits::branchless::avx2_words64_to_vec64_with_uniform(
                                            init_w,
                                            init_is_uniform,
                                        );
                                        vi_vecs = Some(v);
                                        v
                                    }
                                };
                                crate::traits::branchless::avx2_coherence_from_diff_vec64(
                                    _mm256_sub_epi8(vi0, va0),
                                    _mm256_sub_epi8(vi1, va1),
                                )
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            let aw0 = anc_w[0];
                            if init_is_uniform & anc_is_uniform {
                                if aw0 == init_w0 {
                                    1.0
                                } else if init_is_byte_uniform && (aw0 == aw0.rotate_right(8)) {
                                    Q7_FULL_COS_REF[init_b0.wrapping_sub(aw0 as u8) as usize]
                                } else {
                                    coherence_from_word_arrays_with_both_uniform_flags_64(
                                        init_w, true, anc_w, true,
                                    )
                                }
                            } else {
                                coherence_from_word_arrays_with_both_uniform_flags_64(
                                    init_w,
                                    init_is_uniform,
                                    anc_w,
                                    anc_is_uniform,
                                )
                            }
                        }
                    } else {
                        0.0_f32
                    };
                    let proximity_gain = branchless_clamp_f32(1.0 + eff_coh, 0.25, 2.0);
                    let pull = branchless_min_f32(
                        branchless_flush_subnormal_f32(base_pull * proximity_gain),
                        MAX_ATTRACTOR_PULL_FORCE,
                    );
                    if pull > 1.0e-12 {
                        let total_w = running_pull + pull;
                        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                        {
                            if running_pull <= 1.0e-12 {
                                if anc_is_uniform {
                                    run_w0 = anc_w[0];
                                    run_is_uniform = true;
                                } else {
                                    vr_vecs = Some((va0, va1));
                                    run_is_uniform = false;
                                }
                            } else {
                                let (mut vr0, mut vr1) = match vr_vecs {
                                    Some(v) if !run_is_uniform => v,
                                    _ => unsafe {
                                        let rz =
                                            core::arch::x86_64::_mm256_set1_epi64x(run_w0 as i64);
                                        (rz, rz)
                                    },
                                };
                                run_is_uniform = false;
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                if s >= 128 {
                                    vr0 = va0;
                                    vr1 = va1;
                                } else if s > 0 {
                                    unsafe {
                                        use core::arch::x86_64::{
                                            _mm256_or_si256, _mm256_sub_epi8, _mm256_testz_si256,
                                        };
                                        let vd0 = _mm256_sub_epi8(va0, vr0);
                                        let vd1 = _mm256_sub_epi8(va1, vr1);
                                        let vd_or = _mm256_or_si256(vd0, vd1);
                                        if _mm256_testz_si256(vd_or, vd_or) == 0 {
                                            (vr0, vr1) =
                                                crate::traits::branchless::avx2_slerp_from_diff_vec64(
                                                    vr0,
                                                    vr1,
                                                    vd0,
                                                    vd1,
                                                    s,
                                                );
                                        }
                                    }
                                }
                                vr_vecs = Some((vr0, vr1));
                            }
                        }
                        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
                        {
                            if running_pull <= 1.0e-12 {
                                run_w = *anc_w;
                                run_is_uniform = anc_is_uniform;
                            } else if !crate::traits::swar_words_equal_64(&run_w, anc_w) {
                                let ratio_f32 = branchless_clamp_f32(
                                    (pull * fast_inv_pos_normal_f32(total_w)) * 128.0 + 0.5,
                                    0.0,
                                    128.0,
                                );
                                let s = (ratio_f32 as u8) as u64;
                                let bias_restore =
                                    (128u64.wrapping_sub(s)).wrapping_mul(0x0101_0101_0101_0101);
                                let (next_w, next_uni) =
                                    crate::traits::branchless::swar_slerp_words_with_uniform_flag_64(
                                        &run_w,
                                        run_is_uniform,
                                        anc_w,
                                        anc_is_uniform,
                                        s,
                                        bias_restore,
                                    );
                                run_w = next_w;
                                run_is_uniform = next_uni;
                            }
                        }
                        running_pull = total_w;
                    }
                }
            }
            idx += 1;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let mut run_w = if running_pull <= 1.0e-12 {
            *init_w
        } else if run_is_uniform {
            [run_w0; 8]
        } else if let Some((vr0, vr1)) = vr_vecs {
            unsafe { crate::traits::branchless::avx2_vec64_to_words64(vr0, vr1) }
        } else {
            [run_w0; 8]
        };

        let rep_scale_u7 = (branchless_clamp_f32(
            branchless_relu_f32(self.repulsion_gain),
            0.0,
            1.0,
        ) * 128.0
            + 0.5) as u8;
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        let run_is_uniform = if rep_scale_u7 > 0 {
            run_is_uniform || is_word_array_uniform_64(&run_w)
        } else {
            run_is_uniform
        };
        if rep_scale_u7 > 0 {
            let rep_w = *self.repulsion_words_ref();
            if words_any_nonzero_64(&rep_w) {
                let rep_s = rep_scale_u7 as u64;
                let rep_bias = (128u64.wrapping_sub(rep_s)).wrapping_mul(0x0101_0101_0101_0101);
                if is_word_array_uniform_64(&rep_w) {
                    let sc0 = swar_q7_scale_le128_with_bias_x8(rep_w[0], rep_s, rep_bias);
                    if sc0 != 0 {
                        if run_is_uniform {
                            run_w = [swar_q7_sub_sat_x8(run_w[0], sc0); 8];
                        } else {
                            run_w[0] = swar_q7_sub_sat_x8(run_w[0], sc0);
                            run_w[1] = swar_q7_sub_sat_x8(run_w[1], sc0);
                            run_w[2] = swar_q7_sub_sat_x8(run_w[2], sc0);
                            run_w[3] = swar_q7_sub_sat_x8(run_w[3], sc0);
                            run_w[4] = swar_q7_sub_sat_x8(run_w[4], sc0);
                            run_w[5] = swar_q7_sub_sat_x8(run_w[5], sc0);
                            run_w[6] = swar_q7_sub_sat_x8(run_w[6], sc0);
                            run_w[7] = swar_q7_sub_sat_x8(run_w[7], sc0);
                        }
                    }
                } else {
                    run_w[0] = swar_q7_sub_sat_x8(
                        run_w[0],
                        swar_q7_scale_le128_with_bias_x8(rep_w[0], rep_s, rep_bias),
                    );
                    run_w[1] = swar_q7_sub_sat_x8(
                        run_w[1],
                        swar_q7_scale_le128_with_bias_x8(rep_w[1], rep_s, rep_bias),
                    );
                    run_w[2] = swar_q7_sub_sat_x8(
                        run_w[2],
                        swar_q7_scale_le128_with_bias_x8(rep_w[2], rep_s, rep_bias),
                    );
                    run_w[3] = swar_q7_sub_sat_x8(
                        run_w[3],
                        swar_q7_scale_le128_with_bias_x8(rep_w[3], rep_s, rep_bias),
                    );
                    run_w[4] = swar_q7_sub_sat_x8(
                        run_w[4],
                        swar_q7_scale_le128_with_bias_x8(rep_w[4], rep_s, rep_bias),
                    );
                    run_w[5] = swar_q7_sub_sat_x8(
                        run_w[5],
                        swar_q7_scale_le128_with_bias_x8(rep_w[5], rep_s, rep_bias),
                    );
                    run_w[6] = swar_q7_sub_sat_x8(
                        run_w[6],
                        swar_q7_scale_le128_with_bias_x8(rep_w[6], rep_s, rep_bias),
                    );
                    run_w[7] = swar_q7_sub_sat_x8(
                        run_w[7],
                        swar_q7_scale_le128_with_bias_x8(rep_w[7], rep_s, rep_bias),
                    );
                }
            }
        }

        let pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (run_w, pull)
    }

    /// Evaluates composite gravitational pull force and 64-head phase shift for `trail` using
    /// **Register-Resident 8-Word SWAR $S^1$ Geodesic Folding** (`0 fdiv`, `0` intermediate
    /// memory reloads of `trail.accumulated_phase_delta`) and branchless SWAR repulsion subtraction.
    #[inline]
    pub fn evaluate_composite_attractor_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);
        let (run_w, pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        store_q7_words_64(composite_phase_out, &run_w);
        pull
    }

    /// Evaluates the circular statistics summary `(pull_force, r_bar, mean_phase_q7, von_mises_kappa)`
    /// of this lane's composite phase field around `trail`. As more coherent sample points are
    /// added to the lane, `r_bar` approaches `1.0` and `von_mises_kappa` increases.
    #[inline]
    #[must_use]
    pub fn evaluate_field_concentration(
        &self,
        trail: &gcso_pointer_trail_t,
    ) -> (f32, f32, gcso_q7_t, f32) {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);
        let (run_w, pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        let (r_bar, mean_q7, kappa) = circular_resultant_from_words_64(&run_w);
        (pull, r_bar, mean_q7, kappa)
    }

    /// Exports this lane's active and crystallized sample points into a compact 8,448-byte
    /// [`AttractorGrowthSnapshot`] (`Pod`) stamped with an IEEE 802.3 CRC32 checksum.
    #[inline]
    pub fn export_growth_snapshot(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        snapshot_out: &mut AttractorGrowthSnapshot,
    ) -> u32 {
        let prev_valid = (snapshot_out.magic == GCSO_SNAPSHOT_MAGIC_U32)
            & (snapshot_out.abi_version == GCSO_ABI_VERSION_HEX)
            & ((snapshot_out.active_anchor_count as usize) <= MAX_ANCHORS_PER_FIELD);
        let prev_clear_end = if prev_valid {
            snapshot_out.active_anchor_count as usize
        } else {
            MAX_ANCHORS_PER_FIELD
        };
        snapshot_out.magic = GCSO_SNAPSHOT_MAGIC_U32;
        snapshot_out.abi_version = GCSO_ABI_VERSION_HEX;
        snapshot_out.global_step_counter = self.step_counter;
        snapshot_out.coupling_matrix_8x8 = DEFAULT_SLOT_COUPLING_MATRIX_8X8;
        snapshot_out.active_domains_mask = 1u32 << (self.domain_id & 7);
        snapshot_out.repulsion_gain = self.repulsion_gain;
        snapshot_out.reserved_header = [0; 8];
        snapshot_out.repulsion_phase_deltas = self.repulsion_phase_deltas;
        snapshot_out.domain_weights = [1.0; MAX_ATTRACTOR_DOMAINS];
        snapshot_out.reserved_meta = [0; 32];
        snapshot_out.total_potential_energy = self.total_potential_energy();

        let init_w = trail_phase_words_ref(reference_trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);
        let (run_w, _pull) = self.evaluate_composite_attractor_force_words(init_w, init_nonzero);
        store_q7_words_64(&mut snapshot_out.persona_phase_profile, &run_w);
        let r_bar = circular_r_bar_from_words_64(&run_w);
        let kappa = branchless_von_mises_kappa(r_bar);
        snapshot_out.composite_r_bar = r_bar;
        snapshot_out.composite_kappa = kappa;

        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut remaining = self.active_count;
        let mut out_idx = 0usize;
        let mut cryst_cnt = 0u32;
        for anchor in &self.anchors[..cap] {
            if remaining == 0 || out_idx >= MAX_ANCHORS_PER_FIELD {
                break;
            }
            if anchor.is_active != 0 {
                remaining -= 1;
                snapshot_out.anchors[out_idx] = CrystallizedAnchorRecord::from_anchor(anchor);
                cryst_cnt = cryst_cnt.wrapping_add(u32::from(anchor.is_crystallized != 0));
                out_idx += 1;
            }
        }
        let clear_end = prev_clear_end.max(out_idx).min(MAX_ANCHORS_PER_FIELD);
        let mut c_idx = out_idx;
        while c_idx < clear_end {
            snapshot_out.anchors[c_idx] = CrystallizedAnchorRecord::EMPTY;
            c_idx += 1;
        }
        snapshot_out.active_anchor_count = out_idx as u32;
        snapshot_out.crystallized_anchor_count = cryst_cnt;
        snapshot_out.finalize_crc32_for_active_anchors(out_idx)
    }

    /// Restores this lane's state from a verified [`AttractorGrowthSnapshot`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`GCSO_ERROR_VERSION_MISMATCH`] if
    /// `snapshot.verify_integrity()` fails.
    #[inline]
    pub fn restore_from_growth_snapshot(
        &mut self,
        snapshot: &AttractorGrowthSnapshot,
    ) -> GcsoResult<()> {
        snapshot.verify_integrity()?;
        self.clear_anchors()?;
        self.step_counter = snapshot.global_step_counter;
        self.repulsion_gain = snapshot.repulsion_gain;
        self.repulsion_phase_deltas = snapshot.repulsion_phase_deltas;

        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let count = (snapshot.active_anchor_count as usize).min(cap);
        let mut max_id = 0u32;
        let mut active_loaded = 0u32;
        for rec in snapshot.anchors.iter().take(count) {
            if rec.is_active != 0 {
                rec.write_to_anchor_in_place(&mut self.anchors[active_loaded as usize], None);
                if rec.id > max_id {
                    max_id = rec.id;
                }
                active_loaded = active_loaded.wrapping_add(1);
            }
        }
        self.active_count = active_loaded;
        self.next_anchor_id = max_id.wrapping_add(1).max(1);
        Ok(())
    }
}

impl AttractorField for AttractorFieldEngine {
    #[inline(always)]
    fn anchor_count(&self) -> u32 {
        self.active_count
    }

    #[inline(always)]
    fn apply_evaporative_decay(&mut self) {
        Self::apply_evaporative_decay(self);
    }

    #[inline(always)]
    fn total_potential_energy(&self) -> f32 {
        Self::total_potential_energy(self)
    }

    #[inline(always)]
    fn pressure_ratio(&self) -> f32 {
        Self::pressure_ratio(self)
    }

    #[inline]
    fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
        let slot_idx = self.displace_weakest_anchor_slot()?;
        let anchor = &self.anchors[slot_idx];
        if anchor.is_active != 0 {
            let anchor_id = anchor.id;
            let existing_score = anchor.compute_energy_score(self.step_counter);
            self.dissipated_energy_total += existing_score;
            self.anchors[slot_idx].deactivate_in_place();
            self.active_count = self.active_count.saturating_sub(1);
            Some(anchor_id)
        } else {
            None
        }
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32> {
        if prompt.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let phases = hash_prompt_to_q7_phases(prompt, self.domain_id);
        self.insert_anchor_internal(
            GCSO_ANCHOR_TYPE_SYSTEM_PROMPT,
            weight,
            None,
            Some(&phases),
            false,
        )
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        if embedding.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.insert_anchor_internal(anchor_type, weight, Some(embedding), None, false)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32> {
        if dim == 0 || vec.len() < dim {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.insert_anchor_internal(anchor_type, 1.0, Some(vec), None, false)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()> {
        if repulsion_deltas.is_empty() || !branchless_is_finite_f32(gain) || gain <= 0.0 {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.repulsion_gain = gain;
        let copy_len = repulsion_deltas.len().min(MAX_ATTENTION_HEADS);
        self.repulsion_phase_deltas[..copy_len]
            .copy_from_slice(&repulsion_deltas.as_slice()[..copy_len]);
        Ok(())
    }

    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        Ok(self.evaluate_composite_attractor_force(trail, composite_phase_out))
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
        let mut remaining = self.active_count;
        if remaining == 0 {
            return Ok(0);
        }
        let cap = (self.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
        let mut crystallized_count = 0u32;
        for anchor in &mut self.anchors[..cap] {
            if anchor.is_active != 0 {
                if (anchor.is_crystallized == 0) & (anchor.stigmergic_density > 2.0) {
                    anchor.is_crystallized = 1;
                    anchor.anchor_type = GCSO_ANCHOR_TYPE_CRYSTALLIZED;
                    anchor.pull_force =
                        branchless_min_f32(anchor.pull_force * 1.5, MAX_ATTRACTOR_PULL_FORCE);
                    crystallized_count = crystallized_count.wrapping_add(1);
                }
                remaining -= 1;
                if remaining == 0 {
                    break;
                }
            }
        }
        Ok(crystallized_count)
    }

    #[inline]
    fn clear_anchors(&mut self) -> GcsoResult<()> {
        let mut remaining = self.active_count;
        if remaining > 0 {
            let cap = self.anchors.len();
            let mut idx = 0usize;
            while idx < cap {
                if self.anchors[idx].is_active != 0 {
                    self.anchors[idx].deactivate_in_place();
                    remaining -= 1;
                    if remaining == 0 {
                        break;
                    }
                }
                idx += 1;
            }
        }
        self.active_count = 0;
        self.repulsion_gain = 0.0;
        self.repulsion_phase_deltas = [0; MAX_ATTENTION_HEADS];
        Ok(())
    }
}

impl PersonaPatcher for AttractorFieldEngine {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
        if patch_data.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut phases = [0i8; MAX_ATTENTION_HEADS];
        let len = patch_data.len();
        if len >= MAX_ATTENTION_HEADS {
            // SAFETY: `patch_data.len() >= 64` and `[u8; 64]` has identical size and alignment (1) as `[gcso_q7_t; 64]`.
            let patch_ref = unsafe {
                &*patch_data
                    .as_ptr()
                    .cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()
            };
            phases = *patch_ref;
        } else {
            for (h, slot) in phases.iter_mut().enumerate() {
                *slot = patch_data[h % len] as gcso_q7_t;
            }
        }
        let _ = self.insert_anchor_internal(
            GCSO_ANCHOR_TYPE_EMBEDDING,
            1.5,
            None,
            Some(&phases),
            false,
        )?;
        Ok(())
    }
}

// ===================================================================
// 6. Unified Phase Field & Dynamic 8-Lane Working Space
// ===================================================================

/// Unified Macro Attractor Space managing 512 total anchors (`8` hardware lanes $\times 64$ anchors)
/// on a single continuous phase field with automatic spillover and zero-hyperparameter proximity gravity.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiAttractorSpace {
    /// Relative lane weights (Cache Line 0: `offset 0..32`, initialized to neutral `1.0` and self-modulated by phase proximity).
    pub domain_weights: [f32; MAX_ATTRACTOR_DOMAINS],
    /// Active working lane bitmask (`offset 32..36`, `0xFF` = all 8 lanes active on the unified field).
    pub active_domains_mask: u32,
    /// Explicit 4-byte alignment padding before 64-bit counters (`offset 36..40`).
    pub reserved_flags: u32,
    /// Global token step counter across the unified phase space (`offset 40..48`).
    pub global_step_counter: u64,
    /// $8 \times 8$ cross-slot coupling bitmatrix (`u64`, `offset 48..56`).
    pub coupling_matrix_8x8: u64,
    /// Explicit padding completing Cache Line 0 (`offset 56..64`).
    pub reserved_padding: [u8; 8],
    /// 8 hardware-aligned working lanes (`offset 64..328768`, `8 * 64 = 512` unified anchors).
    pub fields: [AttractorFieldEngine; MAX_ATTRACTOR_DOMAINS],
}

// SAFETY: `MultiAttractorSpace` is `#[repr(C, align(64))]` where all fields are `Pod`.
unsafe impl Pod for MultiAttractorSpace {}

impl Default for MultiAttractorSpace {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl MultiAttractorSpace {
    /// Creates a new `MultiAttractorSpace` where all 8 lanes share full `MAX_ANCHORS_PER_FIELD = 64`
    /// capacity (`512` unified anchors) and neutral `1.0` baseline weights (zero manual hyperparameters).
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        let cap = MAX_ANCHORS_PER_FIELD as u32;
        Self {
            fields: [
                AttractorFieldEngine::new(DOMAIN_SYSTEM_BASELINE, cap),
                AttractorFieldEngine::new(DOMAIN_PERSONA, cap),
                AttractorFieldEngine::new(DOMAIN_KNOWLEDGE, cap),
                AttractorFieldEngine::new(DOMAIN_PHASE_REPULSION, cap),
                AttractorFieldEngine::new(DOMAIN_USER_INTERACTION, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_5, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_6, cap),
                AttractorFieldEngine::new(DOMAIN_CUSTOM_7, cap),
            ],
            domain_weights: [1.0_f32; MAX_ATTRACTOR_DOMAINS],
            active_domains_mask: 0xFF,
            reserved_flags: 0,
            global_step_counter: 0,
            coupling_matrix_8x8: DEFAULT_SLOT_COUPLING_MATRIX_8X8,
            reserved_padding: [0; 8],
        }
    }

    /// Resets all 8 domain lanes, weights, active masks, step counters, and coupling matrix in-place
    /// without allocating a temporary `320 KB` `MultiAttractorSpace` frame on the stack.
    #[inline]
    pub fn reset_all_lanes(&mut self) {
        let mut i = 0usize;
        while i < MAX_ATTRACTOR_DOMAINS {
            let field = &mut self.fields[i];
            let mut rem = field.active_count;
            if rem > 0 {
                let mut a = 0usize;
                while a < MAX_ANCHORS_PER_FIELD {
                    if field.anchors[a].is_active != 0 {
                        field.anchors[a].deactivate_in_place();
                        rem -= 1;
                        if rem == 0 {
                            break;
                        }
                    }
                    a += 1;
                }
            }
            field.active_count = 0;
            field.repulsion_gain = 0.0;
            field.repulsion_phase_deltas = [0; MAX_ATTENTION_HEADS];
            field.accumulator_pressure = 0.5;
            field.evaporation_rate = 0.01;
            field.dissipated_energy_total = 0.0;
            field.next_anchor_id = 1;
            field.step_counter = 0;
            self.domain_weights[i] = 1.0_f32;
            i += 1;
        }
        self.active_domains_mask = 0xFF;
        self.reserved_flags = 0;
        self.global_step_counter = 0;
        self.coupling_matrix_8x8 = DEFAULT_SLOT_COUPLING_MATRIX_8X8;
    }

    /// Returns a reference to the System Baseline working lane (Lane 0).
    #[inline(always)]
    #[must_use]
    pub const fn system_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_SYSTEM_BASELINE as usize]
    }

    /// Returns a mutable reference to the System Baseline working lane (Lane 0).
    #[inline(always)]
    pub fn system_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_SYSTEM_BASELINE as usize]
    }

    /// Returns a reference to the Persona working lane (Lane 1).
    #[inline(always)]
    #[must_use]
    pub const fn persona_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_PERSONA as usize]
    }

    /// Returns a mutable reference to the Persona working lane (Lane 1).
    #[inline(always)]
    pub fn persona_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_PERSONA as usize]
    }

    /// Returns a reference to the Knowledge working lane (Lane 2).
    #[inline(always)]
    #[must_use]
    pub const fn knowledge_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_KNOWLEDGE as usize]
    }

    /// Returns a mutable reference to the Knowledge working lane (Lane 2).
    #[inline(always)]
    pub fn knowledge_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_KNOWLEDGE as usize]
    }

    /// Returns a reference to the Phase Repulsion working lane (Lane 3).
    #[inline(always)]
    #[must_use]
    pub const fn repulsion_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_PHASE_REPULSION as usize]
    }

    /// Returns a mutable reference to the Phase Repulsion working lane (Lane 3).
    #[inline(always)]
    pub fn repulsion_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_PHASE_REPULSION as usize]
    }

    /// Returns a reference to the User Interaction working lane (Lane 4).
    #[inline(always)]
    #[must_use]
    pub const fn user_field(&self) -> &AttractorFieldEngine {
        &self.fields[DOMAIN_USER_INTERACTION as usize]
    }

    /// Returns a mutable reference to the User Interaction working lane (Lane 4).
    #[inline(always)]
    pub fn user_field_mut(&mut self) -> &mut AttractorFieldEngine {
        &mut self.fields[DOMAIN_USER_INTERACTION as usize]
    }

    /// Returns the total number of active anchors across the entire 512-anchor unified field.
    #[inline]
    #[must_use]
    pub fn total_unified_anchor_count(&self) -> u32 {
        let mut total = 0u32;
        for field in &self.fields {
            total = total.wrapping_add(field.active_count);
        }
        total
    }

    /// Selects the optimal target working lane (`0..=7`) for a new anchor on the unified field:
    /// uses `preferred_lane` if it has free capacity, or automatically spills over to the active
    /// non-repulsion lane with the lowest pressure / lowest displaceable anchor energy.
    #[inline]
    #[must_use]
    pub fn select_spillover_lane(&self, preferred_lane: u32) -> usize {
        let pref = (preferred_lane as usize) & (MAX_ATTRACTOR_DOMAINS - 1);
        if !self.fields[pref].is_saturated() {
            return pref;
        }
        let mut best_lane = pref;
        let mut lowest_pressure = f32::MAX;
        for (idx, field) in self.fields.iter().enumerate() {
            if idx == (DOMAIN_PHASE_REPULSION as usize) && pref != (DOMAIN_PHASE_REPULSION as usize)
            {
                continue;
            }
            let score = field.pressure_ratio() + field.min_displaceable_energy();
            if score < lowest_pressure {
                lowest_pressure = score;
                best_lane = idx;
            }
        }
        best_lane
    }

    /// Inserts an anchor into the unified 512-anchor phase field, automatically spilling over into
    /// adjacent working lanes if `preferred_lane` is saturated. Returns `(assigned_lane_id, anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `weight <= 0.0` or non-finite, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if all 512 slots resist displacement under pressure.
    #[inline]
    pub fn insert_unified_anchor(
        &mut self,
        preferred_lane: u32,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
        embedding: Option<&AlignedSlice32<'_, f32>>,
    ) -> GcsoResult<(u32, u32)> {
        let lane = self.select_spillover_lane(preferred_lane);
        self.active_domains_mask |= 1 << lane;
        let anchor_id = self.fields[lane].insert_anchor_internal(
            anchor_type,
            weight,
            embedding,
            None,
            false,
        )?;
        Ok((lane as u32, anchor_id))
    }

    /// Registers any [`EpisodicPhaseSample`] into the unified 512-anchor field using its
    /// `preferred_working_lane()` with automatic spillover. Returns `(assigned_lane_id, anchor_id)`.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `sample.sample_mass() <= 0.0`, or
    /// [`GCSO_ERROR_ACTION_HUB_FULL`] if all 512 slots resist displacement.
    #[inline]
    pub fn register_episodic_sample<S: EpisodicPhaseSample + ?Sized>(
        &mut self,
        sample: &S,
        is_crystallized: bool,
    ) -> GcsoResult<(u32, u32)> {
        let lane = self.select_spillover_lane(sample.preferred_working_lane());
        self.active_domains_mask |= 1 << lane;
        let anchor_id = self.fields[lane].add_episodic_sample_anchor(sample, is_crystallized)?;
        Ok((lane as u32, anchor_id))
    }

    /// Autonomously updates `self.domain_weights` from each lane's actual phase resonance and
    /// potential energy relative to `trail` in division-free (`0 fdiv`) mode via
    /// [`branchless_inv_pos_f32`], eliminating manual hyperparameter tuning.
    #[inline]
    pub fn rebalance_weights_from_trail(
        &mut self,
        trail: &gcso_pointer_trail_t,
    ) -> [f32; MAX_ATTRACTOR_DOMAINS] {
        let mut pulls = [0.0_f32; MAX_ATTRACTOR_DOMAINS];
        let mut max_pull = 0.0_f32;
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);

        for (idx, field) in self.fields.iter().enumerate() {
            if ((self.active_domains_mask & (1 << idx)) != 0) & (field.active_count > 0) {
                let (_w, p) = field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
                pulls[idx] = p;
                max_pull = branchless_select_f32(p > max_pull, p, max_pull);
            }
        }

        let has_max = max_pull > 1.0e-6;
        let inv_max = branchless_inv_pos_f32(branchless_select_f32(has_max, max_pull, 1.0));
        for (idx, &p) in pulls.iter().enumerate() {
            let norm_w = branchless_clamp_f32(0.25 + 1.75 * (p * inv_max), 0.25, 2.0);
            self.domain_weights[idx] = branchless_select_f32(has_max & (p > 0.0), norm_w, 1.0);
        }
        self.domain_weights
    }

    /// Projects the unified 512-anchor phase field around `trail` into the 8 hardware SIMD working
    /// lanes (`bank_out: &mut [[gcso_q7_t; 64]; 8]`) and outputs each lane's resonant pull force
    /// (`lane_pulls_out: &mut [f32; 8]`), applying mutual inhibition from Lane 3 (`Repulsion`).
    #[inline]
    pub fn project_active_field_bank_8(
        &self,
        trail: &gcso_pointer_trail_t,
        bank_out: &mut [[gcso_q7_t; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS],
        lane_pulls_out: &mut [f32; MAX_ATTRACTOR_DOMAINS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);

        let rep_field = &self.fields[DOMAIN_PHASE_REPULSION as usize];
        let rep_w_gain = branchless_relu_f32(self.domain_weights[DOMAIN_PHASE_REPULSION as usize]);
        let (rep_words, rep_pull) = if ((rep_field.active_count > 0)
            | (rep_field.repulsion_gain > 0.0))
            & (rep_w_gain > 0.0)
        {
            let (rw, rp) = rep_field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            (rw, rp * rep_w_gain)
        } else {
            ([0u64; 8], 0.0_f32)
        };
        let mut total_pull = 0.0_f32;
        for (idx, field) in self.fields.iter().enumerate() {
            let enabled = (self.active_domains_mask & (1 << idx)) != 0;
            let w = branchless_relu_f32(self.domain_weights[idx]);
            if (!enabled) | (field.active_count == 0) | (w <= 0.0) {
                bank_out[idx] = [0i8; MAX_ATTENTION_HEADS];
                lane_pulls_out[idx] = 0.0;
                continue;
            }
            let (mut slot_w, raw_pull) =
                field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            let mut pull = raw_pull * w;
            if (idx != (DOMAIN_PHASE_REPULSION as usize)) & (rep_pull > 1.0e-6) {
                pull = branchless_q7_inhibit_words_64(&mut slot_w, pull, &rep_words, 0.5);
            }
            store_q7_words_64(&mut bank_out[idx], &slot_w);
            lane_pulls_out[idx] = pull;
            total_pull += pull;
        }
        branchless_min_f32(
            branchless_flush_subnormal_f32(total_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        )
    }

    /// Advances global step counter across all working lanes and evaluates evaporative decay
    /// (touching only Cache Line 0 of empty lanes and running anchor decay only when `active_count > 0`).
    #[inline]
    pub fn advance_step(&mut self) {
        let next_step = self.global_step_counter.wrapping_add(1);
        self.global_step_counter = next_step;
        for field in &mut self.fields {
            field.step_counter = next_step;
            if field.active_count != 0 {
                field.apply_evaporative_decay();
            }
        }
    }

    /// Sets relative lane weighting scalar (`weight >= 0.0`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `weight` is negative/non-finite.
    #[inline]
    pub fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS
            || !branchless_is_finite_f32(weight)
            || weight < 0.0
        {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.domain_weights[domain_id as usize] = weight;
        Ok(())
    }

    /// Enables or disables a specific working lane (`0..=7`) via bitmask.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let bit = 1u32 << idx;
        let mask = (enabled as u32).wrapping_neg();
        self.active_domains_mask = (self.active_domains_mask & !bit) | (bit & mask);
        Ok(())
    }

    /// Retrieves a mutable reference to a specific working lane by ID (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn domain_mut(&mut self, domain_id: u32) -> GcsoResult<&mut AttractorFieldEngine> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(&mut self.fields[idx])
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Retrieves an immutable reference to a specific working lane by ID (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn domain(&self, domain_id: u32) -> GcsoResult<&AttractorFieldEngine> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(&self.fields[idx])
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Exports accumulator pressure metrics for a specific working lane (`0..=7`).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8`.
    #[inline]
    pub fn get_accumulator_summary(&self, domain_id: u32) -> GcsoResult<DomainAccumulatorSummary> {
        let idx = domain_id as usize;
        if idx < MAX_ATTRACTOR_DOMAINS {
            Ok(self.fields[idx].accumulator_summary(self.domain_weights[idx]))
        } else {
            Err(GCSO_ERROR_INVALID_ARGUMENT)
        }
    }

    /// Exports accumulator pressure summaries across all 8 working lanes.
    #[inline]
    #[must_use]
    pub fn get_all_accumulator_summaries(
        &self,
    ) -> [DomainAccumulatorSummary; MAX_ATTRACTOR_DOMAINS] {
        let mut summaries = [DomainAccumulatorSummary::default(); MAX_ATTRACTOR_DOMAINS];
        for (idx, field) in self.fields.iter().enumerate() {
            summaries[idx] = field.accumulator_summary(self.domain_weights[idx]);
        }
        summaries
    }

    /// Identifies the dominant working lane (`0..=7`) currently holding maximum potential energy.
    #[inline]
    #[must_use]
    pub fn find_dominant_domain(&self) -> (u32, f32) {
        let mut max_energy = -1.0_f32;
        let mut dominant_id = DOMAIN_SYSTEM_BASELINE;

        for (idx, field) in self.fields.iter().enumerate() {
            if ((self.active_domains_mask & (1 << idx)) == 0) | (field.active_count == 0) {
                continue;
            }
            let weight = branchless_relu_f32(self.domain_weights[idx]);
            if weight <= 0.0 {
                continue;
            }
            let weighted_energy = field.total_potential_energy() * weight;
            if weighted_energy > max_energy {
                max_energy = weighted_energy;
                dominant_id = idx as u32;
            }
        }
        if max_energy < 0.0 {
            (DOMAIN_SYSTEM_BASELINE, 0.0)
        } else {
            (dominant_id, max_energy)
        }
    }

    /// Register-resident core evaluator for unified multi-domain superposition returning
    /// `(composite_words_64, pull_force)` directly in 8 `u64` SWAR registers.
    #[inline]
    #[must_use]
    pub(crate) fn evaluate_multi_domain_superposition_words(
        &self,
        init_w: &[u64; 8],
        init_nonzero: bool,
    ) -> ([u64; 8], f32) {
        let rep_field = &self.fields[DOMAIN_PHASE_REPULSION as usize];
        let rep_w_gain = branchless_relu_f32(self.domain_weights[DOMAIN_PHASE_REPULSION as usize]);
        let (rep_words, rep_pull) = if ((rep_field.active_count > 0)
            | (rep_field.repulsion_gain > 0.0))
            & (rep_w_gain > 0.0)
        {
            let (rw, rp) = rep_field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            (rw, rp * rep_w_gain)
        } else {
            ([0u64; 8], 0.0_f32)
        };

        let mut running_w = [0u64; 8];
        let mut running_pull = 0.0_f32;
        let mut primary_slot = 0u32;

        for (idx, field) in self.fields.iter().enumerate() {
            if idx == (DOMAIN_PHASE_REPULSION as usize) {
                continue;
            }
            let enabled = (self.active_domains_mask & (1 << idx)) != 0;
            let w = branchless_relu_f32(self.domain_weights[idx]);
            if (!enabled) | (field.active_count == 0) | (w <= 0.0) {
                continue;
            }
            let (mut slot_w, raw_pull) =
                field.evaluate_composite_attractor_force_words(init_w, init_nonzero);
            let mut pull = raw_pull * w;
            if rep_pull > 1.0e-6 {
                pull = branchless_q7_inhibit_words_64(&mut slot_w, pull, &rep_words, 0.5);
            }
            if pull <= 0.0 {
                continue;
            }
            if running_pull <= 0.0 {
                running_w = slot_w;
                running_pull = pull;
                primary_slot = idx as u32;
            } else {
                let coupled =
                    is_slot_pair_coupled(self.coupling_matrix_8x8, primary_slot, idx as u32);
                let syn_gain = branchless_select_f32(coupled, 0.35, 0.0);
                let (fused_w, new_pull, _coh) = branchless_q7_synergy_fuse_words_64(
                    &running_w,
                    running_pull,
                    &slot_w,
                    pull,
                    syn_gain,
                );
                running_w = fused_w;
                running_pull = new_pull;
            }
        }

        if running_pull <= 0.0 {
            return (*init_w, 0.0);
        }

        let pull = branchless_min_f32(
            branchless_flush_subnormal_f32(running_pull),
            MAX_ATTRACTOR_PULL_FORCE,
        );
        (running_w, pull)
    }

    /// Evaluates unified field superposition of gravitational pulls and non-linear cross-slot
    /// phase resonance (`branchless_q7_synergy_fuse_64`) in zero-allocation mode.
    #[inline]
    pub fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        let init_w = trail_phase_words_ref(trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);
        let (running_w, pull) =
            self.evaluate_multi_domain_superposition_words(init_w, init_nonzero);
        store_q7_words_64(composite_phase_out, &running_w);
        pull
    }

    /// Exports the grown persona profile, circular concentration $\kappa_c$, lane weights, and up to
    /// `MAX_ANCHORS_PER_FIELD` (`64`) crystallized/active sample points across the unified space into
    /// a compact 8,448-byte [`AttractorGrowthSnapshot`] (`Pod`) stamped with an IEEE 802.3 CRC32 checksum.
    #[inline]
    pub fn export_growth_snapshot(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        snapshot_out: &mut AttractorGrowthSnapshot,
    ) -> u32 {
        let prev_valid = (snapshot_out.magic == GCSO_SNAPSHOT_MAGIC_U32)
            & (snapshot_out.abi_version == GCSO_ABI_VERSION_HEX)
            & ((snapshot_out.active_anchor_count as usize) <= MAX_ANCHORS_PER_FIELD);
        let prev_clear_end = if prev_valid {
            snapshot_out.active_anchor_count as usize
        } else {
            MAX_ANCHORS_PER_FIELD
        };
        snapshot_out.magic = GCSO_SNAPSHOT_MAGIC_U32;
        snapshot_out.abi_version = GCSO_ABI_VERSION_HEX;
        snapshot_out.global_step_counter = self.global_step_counter;
        snapshot_out.coupling_matrix_8x8 = self.coupling_matrix_8x8;
        snapshot_out.active_domains_mask = self.active_domains_mask;
        snapshot_out.domain_weights = self.domain_weights;
        snapshot_out.repulsion_gain = self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_gain;
        snapshot_out.reserved_header = [0; 8];
        snapshot_out.repulsion_phase_deltas =
            self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_phase_deltas;
        snapshot_out.reserved_meta = [0; 32];
        snapshot_out.total_potential_energy = self.total_potential_energy();

        let init_w = trail_phase_words_ref(reference_trail);
        let init_nonzero = (init_w[0] != 0) || words_any_nonzero_64(init_w);
        let (run_w, _pull) = self.evaluate_multi_domain_superposition_words(init_w, init_nonzero);
        store_q7_words_64(&mut snapshot_out.persona_phase_profile, &run_w);
        let r_bar = circular_r_bar_from_words_64(&run_w);
        let kappa = branchless_von_mises_kappa(r_bar);
        snapshot_out.composite_r_bar = r_bar;
        snapshot_out.composite_kappa = kappa;

        let mut out_idx = 0usize;
        let mut cryst_cnt = 0u32;
        let mut transient_coords = [(0u8, 0u8); MAX_ANCHORS_PER_FIELD];
        let mut transient_len = 0usize;

        // Single pass across active fields (`field.active_count > 0`), stopping as soon as all
        // active anchors in each lane are visited (`0` scans over empty slots).
        for (f_idx, field) in self.fields.iter().enumerate() {
            let mut rem = field.active_count;
            if rem == 0 {
                continue;
            }
            let cap = (field.capacity as usize).min(MAX_ANCHORS_PER_FIELD);
            for (a_idx, anchor) in field.anchors[..cap].iter().enumerate() {
                if rem == 0 {
                    break;
                }
                if anchor.is_active != 0 {
                    rem -= 1;
                    if anchor.is_crystallized != 0 {
                        if out_idx < MAX_ANCHORS_PER_FIELD {
                            snapshot_out.anchors[out_idx] =
                                CrystallizedAnchorRecord::from_anchor(anchor);
                            cryst_cnt = cryst_cnt.wrapping_add(1);
                            out_idx += 1;
                        }
                    } else if transient_len < MAX_ANCHORS_PER_FIELD {
                        transient_coords[transient_len] = (f_idx as u8, a_idx as u8);
                        transient_len += 1;
                    }
                }
            }
        }
        let mut t_i = 0usize;
        while t_i < transient_len && out_idx < MAX_ANCHORS_PER_FIELD {
            let (f_idx, a_idx) = transient_coords[t_i];
            let anchor = &self.fields[f_idx as usize].anchors[a_idx as usize];
            snapshot_out.anchors[out_idx] = CrystallizedAnchorRecord::from_anchor(anchor);
            out_idx += 1;
            t_i += 1;
        }
        let clear_end = prev_clear_end.max(out_idx).min(MAX_ANCHORS_PER_FIELD);
        let mut c_idx = out_idx;
        while c_idx < clear_end {
            snapshot_out.anchors[c_idx] = CrystallizedAnchorRecord::EMPTY;
            c_idx += 1;
        }

        snapshot_out.active_anchor_count = out_idx as u32;
        snapshot_out.crystallized_anchor_count = cryst_cnt;
        snapshot_out.finalize_crc32_for_active_anchors(out_idx)
    }

    /// Restores the unified `MultiAttractorSpace` from a verified [`AttractorGrowthSnapshot`].
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_CONTAINER_CORRUPTED`] or [`GCSO_ERROR_VERSION_MISMATCH`] if
    /// `snapshot.verify_integrity()` fails.
    #[inline]
    pub fn restore_from_growth_snapshot(
        &mut self,
        snapshot: &AttractorGrowthSnapshot,
    ) -> GcsoResult<()> {
        snapshot.verify_integrity()?;
        self.clear_all_domains()?;
        self.global_step_counter = snapshot.global_step_counter;
        self.coupling_matrix_8x8 = snapshot.coupling_matrix_8x8;
        self.active_domains_mask = snapshot.active_domains_mask;
        self.domain_weights = snapshot.domain_weights;
        for field in &mut self.fields {
            field.step_counter = snapshot.global_step_counter;
        }
        self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_gain = snapshot.repulsion_gain;
        self.fields[DOMAIN_PHASE_REPULSION as usize].repulsion_phase_deltas =
            snapshot.repulsion_phase_deltas;

        let count = (snapshot.active_anchor_count as usize).min(MAX_ANCHORS_PER_FIELD);
        for rec in snapshot.anchors.iter().take(count) {
            if rec.is_active == 0 {
                continue;
            }
            let lane = self.select_spillover_lane(rec.domain_id);
            let field = &mut self.fields[lane];
            let slot_idx = field.allocate_anchor_slot(rec.weight.max(0.1))?;
            rec.write_to_anchor_in_place(&mut field.anchors[slot_idx], Some(lane as u32));
            if rec.id >= field.next_anchor_id {
                field.next_anchor_id = rec.id.wrapping_add(1).max(1);
            }
        }
        Ok(())
    }

    /// Serializes a compact 8,448-byte [`AttractorGrowthSnapshot`] directly into `dst_bytes` at `offset`.
    ///
    /// Returns the number of bytes written (`size_of::<AttractorGrowthSnapshot>() = 8448`).
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`] if `dst_bytes.len() < offset + 8448`.
    #[inline]
    pub fn serialize_growth_snapshot_to_bytes(
        &self,
        reference_trail: &gcso_pointer_trail_t,
        dst_bytes: &mut [u8],
        offset: usize,
    ) -> GcsoResult<usize> {
        if let Ok(snap_mut) = AttractorGrowthSnapshot::from_bytes_mut_at(dst_bytes, offset) {
            let _ = self.export_growth_snapshot(reference_trail, snap_mut);
            return Ok(size_of::<AttractorGrowthSnapshot>());
        }
        let mut snap = AttractorGrowthSnapshot::new();
        let _ = self.export_growth_snapshot(reference_trail, &mut snap);
        snap.write_unaligned_to(dst_bytes, offset)?;
        Ok(size_of::<AttractorGrowthSnapshot>())
    }

    /// Zero-copy casts, verifies CRC32, and restores `self` from a 64-byte-aligned `.gcso` ZIMMS
    /// byte buffer at `offset`, returning the zero-copy `&'a AttractorGrowthSnapshot` reference.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`], [`crate::abi::GCSO_ERROR_MISALIGNED_POINTER`],
    /// [`GCSO_ERROR_CONTAINER_CORRUPTED`], or [`GCSO_ERROR_VERSION_MISMATCH`] on failure.
    #[inline]
    pub fn restore_from_snapshot_bytes_zero_copy<'a>(
        &mut self,
        src_bytes: &'a [u8],
        offset: usize,
    ) -> GcsoResult<&'a AttractorGrowthSnapshot> {
        let snap_ref = AttractorGrowthSnapshot::from_bytes_verified_at(src_bytes, offset)?;
        self.restore_from_growth_snapshot(snap_ref)?;
        Ok(snap_ref)
    }

    /// Restores `self` from a potentially unaligned `.gcso` byte slice at `offset` after verifying
    /// magic, ABI version, and IEEE 802.3 CRC32 integrity.
    ///
    /// # Errors
    /// Returns [`crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL`], [`GCSO_ERROR_CONTAINER_CORRUPTED`],
    /// or [`GCSO_ERROR_VERSION_MISMATCH`] on failure.
    #[inline]
    pub fn restore_from_snapshot_bytes_unaligned(
        &mut self,
        src_bytes: &[u8],
        offset: usize,
    ) -> GcsoResult<AttractorGrowthSnapshot> {
        if let Ok(snap_ref) = AttractorGrowthSnapshot::from_bytes_verified_at(src_bytes, offset) {
            self.restore_from_growth_snapshot(snap_ref)?;
            return Ok(*snap_ref);
        }
        let snap = AttractorGrowthSnapshot::read_unaligned_from(src_bytes, offset)?;
        self.restore_from_growth_snapshot(&snap)?;
        Ok(snap)
    }

    /// Adds a system prompt anchor to `domain_id`, spilling over to another lane if full.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `prompt` is empty.
    #[inline]
    pub fn add_system_prompt_to_domain(
        &mut self,
        domain_id: u32,
        prompt: &str,
        weight: f32,
    ) -> GcsoResult<u32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS || prompt.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let lane = self.select_spillover_lane(domain_id);
        self.active_domains_mask |= 1 << lane;
        self.fields[lane].add_system_prompt_anchor(prompt, weight)
    }

    /// Adds an embedding anchor to `domain_id`, spilling over to another lane if full.
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `domain_id >= 8` or `embedding` is empty.
    #[inline]
    pub fn add_embedding_to_domain(
        &mut self,
        domain_id: u32,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        if (domain_id as usize) >= MAX_ATTRACTOR_DOMAINS || embedding.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let (_lane, id) =
            self.insert_unified_anchor(domain_id, anchor_type, weight, Some(embedding))?;
        Ok(id)
    }

    /// Registers a knowledge source embedding into the unified field (preferring Lane 2).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `embedding` is empty or `source_weight <= 0.0`.
    #[inline]
    pub fn register_knowledge_source_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        source_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_KNOWLEDGE,
            embedding,
            GCSO_ANCHOR_TYPE_TOPOLOGICAL,
            source_weight,
        )
    }

    /// Registers a persona modulation patch into the unified field (preferring Lane 1).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `persona_vector` is empty or `persona_weight <= 0.0`.
    #[inline]
    pub fn register_persona_patch_anchor(
        &mut self,
        persona_vector: &AlignedSlice32<'_, f32>,
        persona_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_PERSONA,
            persona_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            persona_weight,
        )
    }

    /// Registers a user interaction context vector into the unified field (preferring Lane 4).
    ///
    /// # Errors
    /// Returns [`GCSO_ERROR_INVALID_ARGUMENT`] if `context_vector` is empty or `context_weight <= 0.0`.
    #[inline]
    pub fn register_user_context_anchor(
        &mut self,
        context_vector: &AlignedSlice32<'_, f32>,
        context_weight: f32,
    ) -> GcsoResult<u32> {
        self.add_embedding_to_domain(
            DOMAIN_USER_INTERACTION,
            context_vector,
            GCSO_ANCHOR_TYPE_EMBEDDING,
            context_weight,
        )
    }

    /// Clears all anchors across all 8 working lanes in the unified space.
    ///
    /// # Errors
    /// Propagates any error from [`AttractorFieldEngine::clear_anchors`].
    #[inline]
    pub fn clear_all_domains(&mut self) -> GcsoResult<()> {
        for field in &mut self.fields {
            field.clear_anchors()?;
        }
        Ok(())
    }
}

impl AttractorField for MultiAttractorSpace {
    #[inline(always)]
    fn anchor_count(&self) -> u32 {
        self.total_unified_anchor_count()
    }

    #[inline(always)]
    fn apply_evaporative_decay(&mut self) {
        self.advance_step();
    }

    #[inline]
    fn total_potential_energy(&self) -> f32 {
        let mut total = 0.0_f32;
        for (idx, field) in self.fields.iter().enumerate() {
            if (self.active_domains_mask & (1 << idx)) != 0 {
                total +=
                    field.total_potential_energy() * branchless_relu_f32(self.domain_weights[idx]);
            }
        }
        branchless_flush_subnormal_f32(total)
    }

    #[inline]
    fn pressure_ratio(&self) -> f32 {
        let total_cap = MAX_ATTRACTOR_DOMAINS * MAX_ANCHORS_PER_FIELD;
        branchless_flush_subnormal_f32(
            (self.total_unified_anchor_count() as f32) * fast_inv_len_f32(total_cap),
        )
    }

    #[inline]
    fn evaporate_weakest_anchor(&mut self) -> Option<u32> {
        let mut weakest_lane = None;
        let mut lowest_e = f32::MAX;
        for (idx, field) in self.fields.iter().enumerate() {
            if field.active_count > 0 {
                let e = field.min_displaceable_energy();
                if e < lowest_e {
                    lowest_e = e;
                    weakest_lane = Some(idx);
                }
            }
        }
        self.fields[weakest_lane?].evaporate_weakest_anchor()
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> GcsoResult<u32> {
        self.add_system_prompt_to_domain(DOMAIN_PERSONA, prompt, weight)
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        embedding: &AlignedSlice32<'_, f32>,
        anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> GcsoResult<u32> {
        let pref = if anchor_type == GCSO_ANCHOR_TYPE_PHASE_REPULSE {
            DOMAIN_PHASE_REPULSION
        } else if anchor_type == GCSO_ANCHOR_TYPE_TOPOLOGICAL {
            DOMAIN_KNOWLEDGE
        } else {
            DOMAIN_PERSONA
        };
        let (_lane, id) = self.insert_unified_anchor(pref, anchor_type, weight, Some(embedding))?;
        Ok(id)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        anchor_type: gcso_anchor_type_t,
        vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> GcsoResult<u32> {
        if dim == 0 || vec.len() < dim {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.add_embedding_anchor(vec, anchor_type, 1.0)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> GcsoResult<()> {
        self.fields[DOMAIN_PHASE_REPULSION as usize].inject_phase_repulsion(repulsion_deltas, gain)
    }

    #[inline]
    fn evaluate_field_force(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        Ok(self.evaluate_multi_domain_superposition(trail, composite_phase_out))
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> GcsoResult<u32> {
        let mut total = 0u32;
        for field in &mut self.fields {
            total = total.wrapping_add(field.aggregate_bottom_up()?);
        }
        Ok(total)
    }

    #[inline]
    fn clear_anchors(&mut self) -> GcsoResult<()> {
        self.clear_all_domains()
    }
}

impl MultiDomainAttractorSpace for MultiAttractorSpace {
    #[inline(always)]
    fn advance_step(&mut self) {
        Self::advance_step(self);
    }

    #[inline(always)]
    fn set_domain_weight(&mut self, domain_id: u32, weight: f32) -> GcsoResult<()> {
        Self::set_domain_weight(self, domain_id, weight)
    }

    #[inline(always)]
    fn set_domain_enabled(&mut self, domain_id: u32, enabled: bool) -> GcsoResult<()> {
        Self::set_domain_enabled(self, domain_id, enabled)
    }

    #[inline(always)]
    fn find_dominant_domain(&self) -> (u32, f32) {
        Self::find_dominant_domain(self)
    }

    #[inline(always)]
    fn evaluate_multi_domain_superposition(
        &self,
        trail: &gcso_pointer_trail_t,
        composite_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> f32 {
        Self::evaluate_multi_domain_superposition(self, trail, composite_phase_out)
    }

    #[inline(always)]
    fn clear_all_domains(&mut self) -> GcsoResult<()> {
        Self::clear_all_domains(self)
    }

    #[inline(always)]
    fn slot_coupling_matrix(&self) -> u64 {
        self.coupling_matrix_8x8
    }

    #[inline]
    fn evaluate_slot_force(
        &self,
        domain_id: u32,
        trail: &gcso_pointer_trail_t,
        slot_phase_out: &mut [gcso_q7_t; MAX_ATTENTION_HEADS],
    ) -> GcsoResult<f32> {
        let idx = domain_id as usize;
        if idx >= MAX_ATTRACTOR_DOMAINS {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        if ((self.active_domains_mask & (1 << idx)) == 0) | (self.fields[idx].active_count == 0) {
            *slot_phase_out = [0i8; MAX_ATTENTION_HEADS];
            return Ok(0.0);
        }
        let pull = self.fields[idx].evaluate_composite_attractor_force(trail, slot_phase_out);
        Ok(pull * branchless_relu_f32(self.domain_weights[idx]))
    }
}

impl PersonaPatcher for MultiAttractorSpace {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> GcsoResult<()> {
        if patch_data.is_empty() {
            return Err(GCSO_ERROR_INVALID_ARGUMENT);
        }
        let mut phases = [0i8; MAX_ATTENTION_HEADS];
        let len = patch_data.len();
        if len >= MAX_ATTENTION_HEADS {
            // SAFETY: `patch_data.len() >= 64` and `[u8; 64]` has identical size and alignment (1) as `[gcso_q7_t; 64]`.
            let patch_ref = unsafe {
                &*patch_data
                    .as_ptr()
                    .cast::<[gcso_q7_t; MAX_ATTENTION_HEADS]>()
            };
            phases = *patch_ref;
        } else {
            for (h, slot) in phases.iter_mut().enumerate() {
                *slot = patch_data[h % len] as gcso_q7_t;
            }
        }
        let lane = self.select_spillover_lane(DOMAIN_PERSONA);
        self.active_domains_mask |= 1 << lane;
        let _ = self.fields[lane].insert_anchor_internal(
            GCSO_ANCHOR_TYPE_EMBEDDING,
            1.5,
            None,
            Some(&phases),
            false,
        )?;
        Ok(())
    }
}

const _: () = {
    assert!(align_of::<AttractorGrowthReport>() == 16);
    assert!(size_of::<AttractorGrowthReport>() == 16);
    assert!(align_of::<CrystallizedAnchorRecord>() == 64);
    assert!(size_of::<CrystallizedAnchorRecord>() == 128);
    assert!(offset_of!(CrystallizedAnchorRecord, last_active_step) == 104);
    assert!(align_of::<AttractorGrowthSnapshot>() == 64);
    assert!(size_of::<AttractorGrowthSnapshot>() == 8448);
    assert!(offset_of!(AttractorGrowthSnapshot, persona_phase_profile) == 64);
    assert!(offset_of!(AttractorGrowthSnapshot, anchors) == 256);
    assert!(align_of::<DomainAccumulatorSummary>() == 16);
    assert!(size_of::<DomainAccumulatorSummary>() == 32);
    assert!(align_of::<AttractorAnchor>() == 64);
    assert!(size_of::<AttractorAnchor>() == 640);
    assert!(offset_of!(AttractorAnchor, phase_deltas) == 64);
    assert!(offset_of!(AttractorAnchor, vector_data) == 128);
    assert!(align_of::<AttractorFieldEngine>() == 64);
    assert!(offset_of!(AttractorFieldEngine, repulsion_phase_deltas) == 64);
    assert!(offset_of!(AttractorFieldEngine, anchors) == 128);
    assert!(align_of::<MultiAttractorSpace>() == 64);
    assert!(offset_of!(MultiAttractorSpace, fields) == 64);
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::aligned::AlignedBuffer64;
    use crate::traits::primitives::DEFAULT_RIPA_CLAMP_MAX_RAD;

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
    fn test_attractor_anchor_energy_and_autonomous_phase_proximity_gravity() {
        let mut anchor_near = AttractorAnchor {
            is_active: 1,
            weight: 1.0,
            pull_force: 1.0,
            stigmergic_density: 1.0,
            last_active_step: 0,
            half_life_steps: 100,
            phase_deltas: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };
        let anchor_opposed = AttractorAnchor {
            phase_deltas: [-96i8; MAX_ATTENTION_HEADS],
            ..anchor_near
        };

        assert!((anchor_near.compute_energy_score(0) - 2.0).abs() < 1e-5);
        assert!((anchor_near.compute_energy_score(100) - 1.0).abs() < 1e-5);

        let trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let (pull_near, coh_near) = anchor_near.compute_resonant_pull(0, &trail);
        let (pull_opp, coh_opp) = anchor_opposed.compute_resonant_pull(0, &trail);
        assert!((coh_near - 1.0).abs() < 1e-4);
        assert!((coh_opp - (-1.0)).abs() < 1e-4);
        assert!(pull_near > pull_opp * 7.0);

        anchor_near.touch(150, 0.5);
        assert_eq!(anchor_near.last_active_step, 150);
        assert!((anchor_near.stigmergic_density - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_accumulator_pressure_evaporative_displacement_and_s1_geodesic_force() {
        let mut engine = AttractorFieldEngine::new(0, 3);
        let buf = AlignedBuffer32::new([0.5f32; 16]);
        let slice = buf.as_aligned_slice();

        let id1 = engine.add_system_prompt_anchor("Anchor 1", 1.0).unwrap();
        let id2 = engine
            .add_embedding_anchor(&slice, GCSO_ANCHOR_TYPE_EMBEDDING, 0.5)
            .unwrap();
        let _id3 = engine.add_system_prompt_anchor("Anchor 3", 2.0).unwrap();

        assert_eq!(engine.active_count(), 3);
        assert!(engine.is_saturated());
        assert_eq!(
            engine.add_system_prompt_anchor("Weak Anchor", 0.01),
            Err(GCSO_ERROR_ACTION_HUB_FULL)
        );

        let id4 = engine
            .add_system_prompt_anchor("Strong Anchor 4", 5.0)
            .unwrap();
        assert_eq!(engine.active_count(), 3);
        assert!(engine.get_anchor(id1).is_some());
        assert!(engine.get_anchor(id2).is_none());
        assert!(engine.get_anchor(id4).is_some());

        engine.anchors[0].stigmergic_density = 3.0;
        assert_eq!(engine.aggregate_bottom_up().unwrap(), 1);
        assert_eq!(engine.crystallized_count(), 1);

        let evaporated = engine.evaporate_weakest_anchor();
        assert!(evaporated.is_some());
        assert_eq!(engine.active_count(), 2);
    }

    #[test]
    fn test_coherence_safe_blanket_steering_and_gcso_pod_snapshot_roundtrip() {
        let mut space = MultiAttractorSpace::new();
        let emb_buf = AlignedBuffer32::new([0.25f32; 16]);
        let emb_slice = emb_buf.as_aligned_slice();

        for _ in 0..80 {
            assert!(space.register_persona_patch_anchor(&emb_slice, 1.0).is_ok());
        }
        assert_eq!(space.total_unified_anchor_count(), 80);
        assert_eq!(space.persona_field().active_count(), 64);

        let mut trail = gcso_pointer_trail_t {
            accumulated_phase_delta: [32i8; MAX_ATTENTION_HEADS],
            ..Default::default()
        };

        let mut bank = [[0i8; MAX_ATTENTION_HEADS]; MAX_ATTRACTOR_DOMAINS];
        let mut lane_pulls = [0.0f32; MAX_ATTRACTOR_DOMAINS];
        let total_pull = space.project_active_field_bank_8(&trail, &mut bank, &mut lane_pulls);
        assert!(total_pull > 0.0);
        assert_eq!(bank[DOMAIN_PERSONA as usize][0], 32);

        let weights = space.rebalance_weights_from_trail(&trail);
        assert!(weights[DOMAIN_PERSONA as usize] >= 1.0);

        let (syn_pull, _coh, neighbors) = space
            .steer_trail_by_coupled_neighborhood(0.5, 0.2, &mut trail)
            .unwrap();
        assert!(syn_pull > 0.0);
        assert!(neighbors >= 1);

        // 1. Verify Blanket EpisodicSampleSteering across diverse sample types (&str, &[u8], [i8; 64], [f32; N], tuples)
        let steerer = DummyPhaseSteerer;
        let mut q_tensor = AlignedBuffer32::new([1.0f32; 16]);
        let str_deltas = "Monika literature club persona"
            .steer_rope_tensors_direct(
                &steerer,
                1,
                0.01,
                DEFAULT_RIPA_CLAMP_MAX_RAD,
                &mut q_tensor.as_aligned_slice_mut(),
                None,
            )
            .unwrap();
        assert!(is_phase_profile_nonzero_64(&str_deltas));

        let raw_q7_sample: [gcso_q7_t; MAX_ATTENTION_HEADS] = [32i8; MAX_ATTENTION_HEADS];
        let q7_deltas = raw_q7_sample.to_rope_phase_deltas(1, 0.01);
        assert_eq!(q7_deltas[0], 32);

        let f32_sample = [0.25f32; 16];
        let f32_deltas = f32_sample.to_rope_phase_deltas(2, 0.01);
        assert_eq!(f32_deltas[0], 32);

        // 2. Verify AttractorGrowthCoordinator batch absorption & concentration growth (kappa_c)
        space.reset_all_lanes();
        let mut growth_trail = gcso_pointer_trail_t::default();
        let mut persona_profile = [0i8; MAX_ATTENTION_HEADS];

        let episodes = [
            ([28i8; MAX_ATTENTION_HEADS], 1.5_f32),
            ([30i8; MAX_ATTENTION_HEADS], 2.0_f32),
            ([32i8; MAX_ATTENTION_HEADS], 2.5_f32),
        ];
        let (report, crystallized_cnt) = space
            .absorb_episodic_sample_batch(
                &episodes,
                1,
                0.8,
                0.7,
                &mut growth_trail,
                &mut persona_profile,
            )
            .unwrap();
        assert!(crystallized_cnt >= 1);
        assert!(report.resultant_r_bar > 0.95);
        assert!(report.von_mises_kappa > 10.0);
        assert!((28..=32).contains(&persona_profile[0]));

        let mut step_rope_deltas = [0i8; MAX_ATTENTION_HEADS];
        let (str_rep, beta_eff) = space
            .absorb_and_steer_rope_step(
                "Monika literature club memory",
                1,
                0.5,
                0.5,
                0.01,
                1.0,
                &mut growth_trail,
                &mut step_rope_deltas,
            )
            .unwrap();
        assert!(str_rep.resonant_pull_force > 0.0);
        assert!(beta_eff > 1.0);

        // 3. Verify .gcso Pod Snapshot (AttractorGrowthSnapshot) serialization & zero-copy restoration
        let expected_unified_count = space.total_unified_anchor_count();
        let mut snap_buf =
            AlignedBuffer64::<u8, { size_of::<AttractorGrowthSnapshot>() }>::zeroed();
        let written = space
            .serialize_growth_snapshot_to_bytes(&growth_trail, snap_buf.as_mut_slice(), 0)
            .unwrap();
        assert_eq!(written, size_of::<AttractorGrowthSnapshot>());

        space.reset_all_lanes();
        let zero_copy_snap = space
            .restore_from_snapshot_bytes_zero_copy(snap_buf.as_slice(), 0)
            .unwrap();
        assert_eq!(zero_copy_snap.magic, GCSO_SNAPSHOT_MAGIC_U32);
        assert!(zero_copy_snap.active_anchor_count >= 1);
        assert!(zero_copy_snap.composite_kappa > 0.0);
        assert_eq!(space.total_unified_anchor_count(), expected_unified_count);
    }
}
