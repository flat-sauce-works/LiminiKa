// tests/ffi/test_abi_layout.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn test_gcso_types_abi_layout_invariants() {
    use std::mem::{align_of, size_of};

    // Verify 128-byte aligned gcso_pointer_trail_t
    assert_eq!(size_of::<liminika_core::abi::gcso_pointer_trail_t>(), 128);
    assert_eq!(align_of::<liminika_core::abi::gcso_pointer_trail_t>(), 128);

    // Verify 64-byte aligned gcso_config_t
    assert_eq!(size_of::<liminika_core::abi::gcso_config_t>(), 64);
    assert_eq!(align_of::<liminika_core::abi::gcso_config_t>(), 16);
}