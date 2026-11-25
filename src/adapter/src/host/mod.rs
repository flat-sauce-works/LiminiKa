// name: src/adapter/src/host/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Domain-agnostic host substrate built strictly over [`liminika_core`] and Rust `std`.
//!
//! Modules under `host/` enforce mechanical vocabulary and import isolation from concrete
//! language-model adapters so working-set paging, container persistence, and future equilibrium
//! reconciliation remain reusable across arbitrary host runtimes.

pub mod pager;

pub use pager::{
    phase_profile_bitcut64, AnchorPagerConfig, AnchorPagerStats, AnchorSlotEntry,
    AnchorWorkingSetPager, PagerStepOutcome, PagerStepQuery, DEFAULT_GRAVITY_DECAY,
    DEFAULT_PREFILTER_TOP_K, DEFAULT_PROMOTION_FLOOR, DEFAULT_WORKING_SET_CAPACITY,
};
