// name: fuzz/fuzz_targets/lmnkp_package.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use liminika_adapter::{
    validate_archive_relative_path, verify_lmnkp_integrity, IntegrityPolicy, LmnkPackageView,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(view) = LmnkPackageView::parse(data) {
        let _ = view.manifest_json();
        let _ = view.compiled_lmnkc();
        let _ = view.verify_integrity(&IntegrityPolicy::AllowUnsigned);
    }
    let _ = verify_lmnkp_integrity(data, &IntegrityPolicy::AllowUnsigned);
    let _ = verify_lmnkp_integrity(data, &IntegrityPolicy::RequireContentHash);
    if let Ok(path) = core::str::from_utf8(data) {
        let _ = validate_archive_relative_path(path);
    }
});
