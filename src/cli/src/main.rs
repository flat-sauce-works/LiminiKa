// src/cli/src/main.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("LiminiKa GCSO Core Engine v0.1.1");

    // 1. Get version string from C-ABI
    let version_str = unsafe {
        let ptr = liminika_core::abi::gcso_abi_get_version_string();
        std::ffi::CStr::from_ptr(ptr).to_str()?
    };
    println!("Loaded C-ABI Engine Version: {}", version_str);

    // 2. Initialize default config via C-ABI
    let mut config = liminika_core::abi::gcso_config_t::default();
    let status = unsafe { liminika_core::abi::gcso_config_init_default(&mut config) };
    assert_eq!(status, liminika_core::abi::GCSO_SUCCESS);

    // 3. Create context instance
    let mut ctx_handle: liminika_core::abi::GcsoContextHandle = std::ptr::null_mut();
    let status = unsafe { liminika_core::abi::gcso_context_create(&config, &mut ctx_handle) };
    assert_eq!(status, liminika_core::abi::GCSO_SUCCESS);

    println!("Context initialized successfully. Ready for token stepping.");

    // 4. Clean up context
    unsafe { liminika_core::abi::gcso_context_destroy(ctx_handle) };
    Ok(())
}
