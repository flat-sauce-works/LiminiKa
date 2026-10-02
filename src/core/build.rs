// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");
    let kernels_dir = repo_root.join("src/kernels");
    let include_dir = repo_root.join("include");

    // Re-run build script if any included headers or C++ kernel sources change
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    // Configure CMake build options based on active Cargo features
    let mut cfg = cmake::Config::new(&kernels_dir);
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_SHARED_LIBS", "OFF");

    #[cfg(feature = "cuda")]
    cfg.define("LIMINIKA_ENABLE_CUDA", "ON");

    #[cfg(feature = "vulkan")]
    cfg.define("LIMINIKA_ENABLE_VULKAN", "ON");

    #[cfg(feature = "metal")]
    cfg.define("LIMINIKA_ENABLE_METAL", "ON");

    let dst = cfg.build();

    // Export linker search paths (supporting both lib and lib64 layout structures)
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Link C++ core kernel library
    println!("cargo:rustc-link-lib=static=liminika_kernels_cpu");

    #[cfg(feature = "cuda")]
    println!("cargo:rustc-link-lib=static=liminika_kernels_cuda");

    #[cfg(feature = "vulkan")]
    println!("cargo:rustc-link-lib=static=liminika_kernels_vulkan");

    #[cfg(feature = "metal")]
    println!("cargo:rustc-link-lib=static=liminika_kernels_metal");

    // Link native C++ runtime library based on target operating system
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "windows" => {
            // MSVC manages C++ runtime linkage implicitly
        }
        _ => {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}
