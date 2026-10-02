// src/core/build.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");

    // Rerun triggers
    println!("cargo:rerun-if-changed=../../include/liminika/");
    println!("cargo:rerun-if-changed=../../src/kernels/");

    // Build C++ Kernels via CMake
    let dst = cmake::Config::new(repo_root.join("src/kernels"))
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=liminika_kernels_cpu");

    // Link C++ Standard Library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");
}