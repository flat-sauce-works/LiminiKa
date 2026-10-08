// name: src/core/build.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::env;
use std::path::PathBuf;
use std::process::Command;

/// Checks whether the CUDA compiler (`nvcc`) or relevant CUDA SDK environment variables exist.
fn is_nvcc_available() -> bool {
    let cuda_env_vars = [
        "CUDA_PATH",
        "CUDA_ROOT",
        "CUDA_TOOLKIT_ROOT_DIR",
        "CUDAToolkit_ROOT",
    ];

    for var in &cuda_env_vars {
        if env::var(var).is_ok() {
            return true;
        }
    }

    Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Checks whether Vulkan SDK or vulkaninfo runtime is available on the target environment.
fn is_vulkan_available() -> bool {
    if env::var("VULKAN_SDK").is_ok() {
        return true;
    }

    // Check system32 vulkan runtime on Windows target
    if cfg!(target_os = "windows") {
        if let Ok(windir) = env::var("WINDIR") {
            let vulkan_dll = PathBuf::from(windir).join("System32").join("vulkan-1.dll");
            if vulkan_dll.exists() {
                return true;
            }
        }
    }

    Command::new("vulkaninfo")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Helper function to parse boolean environment variables.
fn parse_bool_env(var_name: &str) -> Option<bool> {
    env::var(var_name).ok().map(|v| {
        let lower = v.to_lowercase();
        lower == "1" || lower == "on" || lower == "true"
    })
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let repo_root = manifest_dir.join("../..");
    let kernels_dir = repo_root.join("src/kernels");
    let include_dir = repo_root.join("include");

    // Target configuration evaluation for cross-compilation safety
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    let is_apple = target_os == "macos" || target_os == "ios";
    let is_windows = target_os == "windows";

    // Track build script and native directory dependencies
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", include_dir.display());
    println!("cargo:rerun-if-changed={}", kernels_dir.display());
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_CUDA");
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_VULKAN");
    println!("cargo:rerun-if-env-changed=LIMINIKA_ENABLE_METAL");

    // Evaluate GPU backend feature flags and env overrides
    let enable_cuda = parse_bool_env("LIMINIKA_ENABLE_CUDA")
        .unwrap_or_else(|| env::var("CARGO_FEATURE_CUDA").is_ok() && is_nvcc_available());

    let enable_vulkan = parse_bool_env("LIMINIKA_ENABLE_VULKAN")
        .unwrap_or_else(|| env::var("CARGO_FEATURE_VULKAN").is_ok() && is_vulkan_available());

    // Metal is strictly restricted to Apple target platforms
    let enable_metal = is_apple
        && parse_bool_env("LIMINIKA_ENABLE_METAL")
            .unwrap_or_else(|| env::var("CARGO_FEATURE_METAL").is_ok() || target_os == "macos");

    // Configure CMake build configuration
    let mut cfg = cmake::Config::new(&kernels_dir);

    // Map Cargo profile to CMake build type
    let cmake_build_type = match profile.as_str() {
        "release" | "bench" => "Release",
        "min-size-rel" => "MinSizeRel",
        "rel-with-deb-info" => "RelWithDebInfo",
        _ => "Debug",
    };

    cfg.define("CMAKE_BUILD_TYPE", cmake_build_type)
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
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

    // Forward target architecture to CMake if cross-compiling
    if !target_arch.is_empty() {
        cfg.define("LIMINIKA_TARGET_ARCH", &target_arch);
    }

    let dst = cfg.build();

    // Export linker search paths across platform generator defaults
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-search=native={}", dst.display());

    // Link CPU core kernel library
    println!("cargo:rustc-link-lib=static=liminika_kernels_cpu");

    // Conditionally link backend acceleration static libraries and SDK frameworks
    if enable_cuda {
        println!("cargo:rustc-link-lib=static=liminika_kernels_cuda");
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib/x64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        }
        println!("cargo:rustc-link-lib=dylib=cudart");
    }

    if enable_vulkan {
        println!("cargo:rustc-link-lib=static=liminika_kernels_vulkan");
        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            println!("cargo:rustc-link-search=native={}/lib", vulkan_sdk);
            if is_windows {
                println!("cargo:rustc-link-search=native={}/Lib", vulkan_sdk);
            }
        }
        println!("cargo:rustc-link-lib=dylib=vulkan");
    }

    if enable_metal && is_apple {
        println!("cargo:rustc-link-lib=static=liminika_kernels_metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
    }

    // Standard C++ runtime linkage strategy per target OS and toolchain
    match target_os.as_str() {
        "macos" | "ios" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "windows" => {
            if target_env == "gnu" {
                println!("cargo:rustc-link-lib=stdc++");
            }
            // MSVC target manages C++ runtime linkage implicitly
        }
        _ => {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}
