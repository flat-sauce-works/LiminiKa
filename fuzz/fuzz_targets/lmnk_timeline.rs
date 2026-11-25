// name: fuzz/fuzz_targets/lmnk_timeline.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use liminika_adapter::LmnkTimeline;

fuzz_target!(|data: &[u8]| {
    let _ = LmnkTimeline::from_bytes(data);
});
