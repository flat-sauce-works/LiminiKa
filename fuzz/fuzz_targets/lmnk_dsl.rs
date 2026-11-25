// name: fuzz/fuzz_targets/lmnk_dsl.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use liminika_dsl::parse_and_validate_lmnk;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = core::str::from_utf8(data) {
        let _ = parse_and_validate_lmnk(text);
    }
});
