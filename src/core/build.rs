// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");
    let kernels_dir = repo_root.join("src/kernels");
    let include_dir = repo_root.join("include");

    // Re-run build script if included headers, sources, or build flags change
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_CUDA");
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_VULKAN");
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_METAL");

    // Evaluate hardware acceleration backend enablement via Cargo features or env vars
    let enable_cuda = cfg!(feature = "cuda")
        || env::var("LIMINIKA_ENABLE_CUDA")
            .map(|v| v == "1" || v == "ON" || v == "true")
            .unwrap_or(false);
    let enable_vulkan = cfg!(feature = "vulkan")
        || env::var("LIMINIKA_ENABLE_VULKAN")
            .map(|v| v == "1" || v == "ON" || v == "true")
            .unwrap_or(false);
    let enable_metal = cfg!(feature = "metal")
        || env::var("LIMINIKA_ENABLE_METAL")
            .map(|v| v == "1" || v == "ON" || v == "true")
            .unwrap_or(false);

    // Configure CMake build options with explicit ON/OFF flags
    let mut cfg = cmake::Config::new(&kernels_dir);
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LIMINIKA_ENABLE_CUDA", if enable_cuda { "ON" } else { "OFF" })
        .define("LIMINIKA_ENABLE_VULKAN", if enable_vulkan { "ON" } else { "OFF" })
        .define("LIMINIKA_ENABLE_METAL", if enable_metal { "ON" } else { "OFF" });

    let dst = cfg.build();

    // Export linker search paths for native target library layouts
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());

    // Link C++ CPU core kernel static library unconditionally
    println!("cargo:rustc-link-lib=static=liminika_kernels_cpu");

    // Conditionally link target acceleration static libraries
    if enable_cuda {
        println!("cargo:rustc-link-lib=static=liminika_kernels_cuda");
    }
    if enable_vulkan {
        println!("cargo:rustc-link-lib=static=liminika_kernels_vulkan");
    }
    if enable_metal {
        println!("cargo:rustc-link-lib=static=liminika_kernels_metal");
    }

    // Link system C++ standard library based on target operating system
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