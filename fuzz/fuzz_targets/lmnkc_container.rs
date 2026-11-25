// name: fuzz/fuzz_targets/lmnkc_container.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use liminika_adapter::{
    verify_lmnkc_integrity, EquilibriumConfig, EquilibriumState, IntegrityPolicy,
};
use liminika_core::runtime::LmnkContainerView;

fuzz_target!(|data: &[u8]| {
    let _ = LmnkContainerView::parse(data);
    let _ = verify_lmnkc_integrity(data, &IntegrityPolicy::AllowUnsigned);
    let _ = verify_lmnkc_integrity(data, &IntegrityPolicy::RequireContentHash);
    let _ = EquilibriumState::from_lmnkc_bytes(data, EquilibriumConfig::default());
});
