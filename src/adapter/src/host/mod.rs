// name: src/adapter/src/host/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-agnostic host substrate built strictly over [`liminika_core`] and Rust `std`.
//!
//! Modules under `host/` enforce mechanical vocabulary and import isolation from concrete
//! language-model adapters so working-set paging, container persistence, and future equilibrium
//! reconciliation remain reusable across arbitrary host runtimes.

pub mod equilibrium;
pub mod integrity;
pub mod pager;
pub mod path_guard;

pub use equilibrium::{
    DEFAULT_EQUILIBRIUM_DECAY, DEFAULT_EVAPORATION_FLOOR, DEFAULT_MAX_BACKING_ANCHORS,
    DEFAULT_MAX_GRAVITY_MASS, DEFAULT_MERGE_COHERENCE_THRESHOLD, DEFAULT_NOVEL_ARRIVAL_EXCITATION,
    DEFAULT_REINFORCEMENT_GAIN, DEFAULT_STEADY_EXCITATION_CAP, EquilibriumConfig, EquilibriumError,
    EquilibriumJoinReport, EquilibriumState, HOST_EQUILIBRIUM_META_SECTION_KIND,
};
pub use integrity::{
    CONTENT_HASH_BYTES, IntegrityError, IntegrityPolicy, IntegrityReport, IntegrityTrustLevel,
    PUBLIC_KEY_BYTES, SECRET_KEY_SEED_BYTES, SIGNATURE_BYTES, SigningKeyPair, VerifyingPublicKey,
    attach_lmnkc_content_hash, compute_archive_entries_content_hash, compute_lmnkc_content_hash,
    decode_hex_exact, encode_hex_lower, sign_archive_content_hash, sign_lmnkc_container,
    verify_archive_entries_integrity, verify_lmnkc_integrity,
};
pub use pager::{
    AnchorPagerConfig, AnchorPagerStats, AnchorSlotEntry, AnchorWorkingSetPager,
    DEFAULT_GRAVITY_DECAY, DEFAULT_PREFILTER_TOP_K, DEFAULT_PROMOTION_FLOOR,
    DEFAULT_WORKING_SET_CAPACITY, PagerStepOutcome, PagerStepQuery, phase_profile_bitcut64,
};
pub use path_guard::{
    MAX_ARCHIVE_PATH_BYTES, PathGuardError, is_windows_reserved_device_name,
    validate_archive_relative_path, validate_local_output_path, validate_windows_style_output_path,
};
