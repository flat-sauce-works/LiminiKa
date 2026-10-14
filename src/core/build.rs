// name: src/core/build.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Build script for the Geometric Cellular Sheaf Orchestrator (GCSO) core engine.
//!
//! Handles CMake initialization for low-level C++ / GPU kernels (CUDA, Metal, Vulkan),
//! registers custom `cfg` flags to prevent compiler linter warnings, and links native static libraries.

#![allow(missing_docs)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// List of valid file extensions tracked for Cargo re-run checking.
const TRACKED_EXTENSIONS: &[&str] = &[
    "h", "hpp", "c", "cpp", "cu", "cuh", "metal", "mm", "glsl", "spv", "txt", "cmake",
];

/// Registers all custom cfg identifiers to prevent `unexpected_cfgs` warnings on Rust 1.80+.
fn register_check_cfgs() {
    let custom_cfgs = [
        "has_kernel_cuda",
        "has_kernel_vulkan",
        "has_kernel_metal",
        "has_simd_avx2",
        "has_simd_avx512",
        "has_simd_neon",
        "has_simd_sve",
        "has_simd_rvv",
        "docsrs",
    ];

    for cfg in &custom_cfgs {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
    }
}

/// Recursively registers valid source and header files in a directory for Cargo re-run tracking,
/// ignoring hidden files, version control folders, and build artifact directories.
fn track_directory_changes(dir: &Path) {
    if !dir.exists() || !dir.is_dir() {
        return;
    }

    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();

            // Ignore hidden files, build output folders, and git directories
            if file_name.starts_with('.') || file_name == "target" || file_name == "build" {
                continue;
            }

            if path.is_dir() {
                track_directory_changes(&path);
            } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                if TRACKED_EXTENSIONS.contains(&ext.to_lowercase().as_str())
                    || file_name == "CMakeLists.txt"
                {
                    println!("cargo:rerun-if-changed={}", path.display());
                }
            }
        }
    }
}

/// Checks whether the CUDA compiler (nvcc) or relevant CUDA SDK environment variables exist.
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
        let lower = v.trim().to_lowercase();
        lower == "1" || lower == "on" || lower == "true" || lower == "yes"
    })
}

/// Helper function to check if a specific target feature is present in the target features list.
fn has_target_feature(target_features: &str, feature: &str) -> bool {
    target_features.split(',').any(|f| f.trim() == feature)
}

/// Resolves default CUDA target compute architectures if not specified by environment variables.
fn resolve_cuda_architectures() -> String {
    if let Ok(arch) = env::var("LIMINIKA_CUDA_ARCH").or_else(|_| env::var("CUDA_ARCHITECTURES")) {
        return arch;
    }
    // Default compute architectures targeting modern GeForce/DataCenter GPUs (Turing, Ampere, Ada, Hopper)
    "75;80;86;89;90".to_string()
}

fn main() {
    // Register expected custom cfgs for compiler diagnostic checking
    register_check_cfgs();

    // Check if building under docs.rs sandbox environment
    if env::var("DOCS_RS").is_ok() {
        println!("cargo:rustc-cfg=docsrs");
        return;
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let repo_root = manifest_dir.join("../..");
    let kernels_dir = repo_root.join("src/kernels");
    let include_dir = repo_root.join("include");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    let is_apple = target_os == "macos"
        || target_os == "ios"
        || target_os == "tvos"
        || target_os == "visionos";
    let is_windows = target_os == "windows";
    let is_msvc = is_windows && target_env == "msvc";

    // Track build script, root CMake file, C headers, and C++/CUDA/Metal source changes
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("CMakeLists.txt").display()
    );
    track_directory_changes(&include_dir);
    track_directory_changes(&kernels_dir);

    // Track environment variables impacting native compilation
    let tracked_envs = [
        "LIMINIKA_ENABLE_CUDA",
        "LIMINIKA_ENABLE_VULKAN",
        "LIMINIKA_ENABLE_METAL",
        "LIMINIKA_CUDA_ARCH",
        "CUDA_ARCHITECTURES",
        "VULKAN_SDK",
        "CUDA_PATH",
        "CC",
        "CXX",
        "CFLAGS",
        "CXXFLAGS",
        "CMAKE_GENERATOR",
        "TARGET",
        "HOST",
    ];
    for var in &tracked_envs {
        println!("cargo:rerun-if-env-changed={var}");
    }

    // Evaluate GPU backend feature flags and environment variable overrides
    let cuda_explicit = parse_bool_env("LIMINIKA_ENABLE_CUDA");
    let enable_cuda = cuda_explicit
        .unwrap_or_else(|| env::var("CARGO_FEATURE_CUDA").is_ok() && is_nvcc_available());

    if cuda_explicit == Some(true) && !is_nvcc_available() {
        println!("cargo:warning=LIMINIKA_ENABLE_CUDA is set to true, but CUDA compiler (nvcc) was not detected.");
    }

    let vulkan_explicit = parse_bool_env("LIMINIKA_ENABLE_VULKAN");
    let enable_vulkan = vulkan_explicit
        .unwrap_or_else(|| env::var("CARGO_FEATURE_VULKAN").is_ok() && is_vulkan_available());

    if vulkan_explicit == Some(true) && !is_vulkan_available() {
        println!("cargo:warning=LIMINIKA_ENABLE_VULKAN is set to true, but Vulkan SDK / runtime was not detected.");
    }

    let metal_explicit = parse_bool_env("LIMINIKA_ENABLE_METAL");
    let enable_metal = is_apple
        && metal_explicit
            .unwrap_or_else(|| env::var("CARGO_FEATURE_METAL").is_ok() || target_os == "macos");

    // Detect CPU SIMD features conditionally according to target architecture to prevent cross-compilation & cross-architecture flag leaking
    let is_x86_64 = target_arch == "x86_64";
    let is_aarch64 = target_arch == "aarch64";
    let is_riscv64 = target_arch == "riscv64";

    let has_avx2 = is_x86_64
        && (env::var("CARGO_FEATURE_AVX2").is_ok()
            || has_target_feature(&target_features, "avx2"));
    let has_avx512 = is_x86_64
        && (env::var("CARGO_FEATURE_AVX512").is_ok()
            || has_target_feature(&target_features, "avx512f")
            || has_target_feature(&target_features, "avx512"));
    let has_neon = is_aarch64
        && (env::var("CARGO_FEATURE_NEON").is_ok()
            || has_target_feature(&target_features, "neon")
            || is_apple);
    let has_sve = is_aarch64
        && !is_apple
        && (env::var("CARGO_FEATURE_SVE").is_ok() || has_target_feature(&target_features, "sve"));
    let has_rvv = is_riscv64
        && (env::var("CARGO_FEATURE_RVV").is_ok() || has_target_feature(&target_features, "v"));

    // Emit custom Rust conditional compilation flags based on backend availability
    if enable_cuda {
        println!("cargo:rustc-cfg=has_kernel_cuda");
    }
    if enable_vulkan {
        println!("cargo:rustc-cfg=has_kernel_vulkan");
    }
    if enable_metal {
        println!("cargo:rustc-cfg=has_kernel_metal");
    }
    if has_avx2 {
        println!("cargo:rustc-cfg=has_simd_avx2");
    }
    if has_avx512 {
        println!("cargo:rustc-cfg=has_simd_avx512");
    }
    if has_neon {
        println!("cargo:rustc-cfg=has_simd_neon");
    }
    if has_sve {
        println!("cargo:rustc-cfg=has_simd_sve");
    }
    if has_rvv {
        println!("cargo:rustc-cfg=has_simd_rvv");
    }

    // Configure CMake build pipeline with C++20 and strict execution flags
    let mut cfg = cmake::Config::new(&kernels_dir);

    let cmake_build_type = match profile.as_str() {
        "release" | "bench" => "Release",
        "min-size-rel" => "MinSizeRel",
        "rel-with-deb-info" => "RelWithDebInfo",
        _ => "Debug",
    };

    cfg.define("CMAKE_BUILD_TYPE", cmake_build_type)
        .define("CMAKE_C_STANDARD", "11")
        .define("CMAKE_C_STANDARD_REQUIRED", "ON")
        .define("CMAKE_CXX_STANDARD", "20")
        .define("CMAKE_CXX_STANDARD_REQUIRED", "ON")
        .define("CMAKE_CXX_EXTENSIONS", "OFF")
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
        )
        .define("LIMINIKA_ENABLE_AVX2", if has_avx2 { "ON" } else { "OFF" })
        .define(
            "LIMINIKA_ENABLE_AVX512",
            if has_avx512 { "ON" } else { "OFF" },
        )
        .define("LIMINIKA_ENABLE_NEON", if has_neon { "ON" } else { "OFF" })
        .define("LIMINIKA_ENABLE_SVE", if has_sve { "ON" } else { "OFF" })
        .define("LIMINIKA_ENABLE_RVV", if has_rvv { "ON" } else { "OFF" });

    // Enforce strict C++ compiler warning and encoding flags based on toolchain and target architecture
    if is_msvc {
        cfg.cxxflag("/W4")
            .cxxflag("/O2")
            .cxxflag("/permissive-")
            .cxxflag("/EHsc")
            .cxxflag("/utf-8")
            .cxxflag("/Zc:preprocessor")
            .cxxflag("/Zc:inline");

        // Conditionally apply architecture-specific SIMD flags for MSVC
        if target_arch == "x86_64" {
            if has_avx512 {
                cfg.cxxflag("/arch:AVX512");
            } else if has_avx2 {
                cfg.cxxflag("/arch:AVX2");
            }
        }
    } else {
        cfg.cxxflag("-Wall")
            .cxxflag("-Wextra")
            .cxxflag("-Wpedantic")
            .cxxflag("-Wconversion")
            .cxxflag("-Wno-unused-parameter")
            .cxxflag("-O3")
            .cxxflag("-fstrict-aliasing");

        // Conditionally apply architecture-specific SIMD flags to prevent cross-compilation errors
        if target_arch == "x86_64" {
            if has_avx512 {
                cfg.cxxflag("-mavx512f")
                    .cxxflag("-mavx512bw")
                    .cxxflag("-mavx512dq");
            } else if has_avx2 {
                cfg.cxxflag("-mavx2").cxxflag("-mfma");
            }
        } else if target_arch == "aarch64" && has_sve {
            cfg.cxxflag("-march=armv8-a+sve");
        }
    }

    // Configure target system parameters when cross-compiling
    if host != target {
        let system_name = match target_os.as_str() {
            "macos" | "ios" | "visionos" | "tvos" => "Darwin",
            "windows" => "Windows",
            "android" => "Android",
            _ => "Linux",
        };
        cfg.define("CMAKE_SYSTEM_NAME", system_name);
        cfg.define("CMAKE_SYSTEM_PROCESSOR", &target_arch);
    }

    // Pass custom CUDA target architectures if enabled
    if enable_cuda {
        let cuda_archs = resolve_cuda_architectures();
        cfg.define("CMAKE_CUDA_ARCHITECTURES", &cuda_archs);
    }

    if !target_arch.is_empty() {
        cfg.define("LIMINIKA_TARGET_ARCH", &target_arch);
    }

    // Configure MSVC runtime library alignment on Windows MSVC toolchains
    if is_msvc {
        cfg.define(
            "CMAKE_MSVC_RUNTIME_LIBRARY",
            "MultiThreaded$<$<CONFIG:Debug>:Debug>",
        );
    }

    let dst = cfg.build();

    // Search all potential output locations across Multi-Config and Single-Config CMake generators
    let search_paths = [
        dst.join("lib"),
        dst.join("lib64"),
        dst.join("build"),
        dst.join("Release"),
        dst.join("Debug"),
        dst.clone(),
    ];

    for path in &search_paths {
        if path.exists() {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
    }

    // Link CPU core kernel static library
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
        let mut vulkan_lib_found = false;

        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            println!("cargo:rustc-link-search=native={}/lib", vulkan_sdk);
            if is_windows {
                println!("cargo:rustc-link-search=native={}/Lib", vulkan_sdk);
            }
            vulkan_lib_found = true;
        }

        if is_windows && !vulkan_lib_found {
            if let Ok(program_files) = env::var("ProgramFiles") {
                let vulkan_base = PathBuf::from(program_files).join("VulkanSDK");
                if vulkan_base.exists() {
                    if let Ok(entries) = fs::read_dir(vulkan_base) {
                        for entry in entries.flatten() {
                            let lib_path = entry.path().join("Lib");
                            if lib_path.exists() {
                                println!("cargo:rustc-link-search=native={}", lib_path.display());
                                vulkan_lib_found = true;
                                break;
                            }
                        }
                    }
                }
            }
        }

        if vulkan_lib_found || !is_windows {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    }

    if enable_metal && is_apple {
        println!("cargo:rustc-link-lib=static=liminika_kernels_metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }

    // Standard C++ runtime linkage strategy per target OS and toolchain
    match target_os.as_str() {
        "macos" | "ios" | "visionos" | "tvos" => {
            println!("cargo:rustc-link-lib=c++");
        }
        "windows" => {
            if target_env == "gnu" {
                println!("cargo:rustc-link-lib=stdc++");
            }
        }
        _ => {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
}