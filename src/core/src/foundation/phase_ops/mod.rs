// name: src/core/src/foundation/phase_ops/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! 64-head SWAR/AVX2 phase-operation algebra (`coherence`, `gravity_fold`, `toroidal_centroid`).

pub mod coherence;
pub mod gravity_fold;
pub mod toroidal_centroid;

pub(crate) use self::coherence::coherence_from_uniform_words_x8;
pub use self::coherence::*;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use self::coherence::{
    avx2_coherence_from_diff_vec64, avx2_hsum_ps, avx2_hsum2_ps, avx2_slerp_from_diff_vec64,
    avx2_slerp_vec64, avx2_vec64_to_words64, avx2_words64_to_vec64,
    avx2_words64_to_vec64_with_uniform,
};
pub use self::gravity_fold::*;
pub(crate) use self::toroidal_centroid::step_toroidal_centroid_words_cached_64;
pub use self::toroidal_centroid::*;
