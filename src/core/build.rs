// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Checks whether the CUDA compiler (`nvcc`) or relevant environment variables are available on the host system.
fn is_nvcc_available() -> bool {
    if env::var("CUDA_PATH").is_ok() || env::var("CUDAToolkit_ROOT").is_ok() {
        return true;
    }
    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

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

    // Evaluate CUDA backend enablement.
    // Prioritize explicit LIMINIKA_ENABLE_CUDA environment variable override.
    // Fallback to cargo feature "cuda" ONLY IF nvcc/CUDA Toolkit is present on the system.
    let enable_cuda = match env::var("LIMINIKA_ENABLE_CUDA") {
        Ok(v) => {
            let lower = v.to_lowercase();
            lower == "1" || lower == "on" || lower == "true"
        }
        Err(_) => cfg!(feature = "cuda") && is_nvcc_available(),
    };

    let enable_vulkan = match env::var("LIMINIKA_ENABLE_VULKAN") {
        Ok(v) => {
            let lower = v.to_lowercase();
            lower == "1" || lower == "on" || lower == "true"
        }
        Err(_) => cfg!(feature = "vulkan"),
    };

    let enable_metal = match env::var("LIMINIKA_ENABLE_METAL") {
        Ok(v) => {
            let lower = v.to_lowercase();
            lower == "1" || lower == "on" || lower == "true"
        }
        Err(_) => cfg!(target_os = "macos") || cfg!(feature = "metal"),
    };

    // Configure CMake build options with explicit ON/OFF flags
    let mut cfg = cmake::Config::new(&kernels_dir);
    cfg.define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define(
            "LIMINIKA_ENABLE_CUDA",
            if enable_cuda { "ON" } else { "OFF" },
        )
        .define(
            "LIMINIKA_ENABLE_VULKAN",
            if enable_vulkan { "ON" } else { "OFF" },
        )
        .define(
            "LIMINIKA_ENABLE_METAL",
            if enable_metal { "ON" } else { "OFF" },
        );

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
