use std::env;

fn main() {
    // Re-run build script if kernel files or headers change
    println!("cargo:rerun-if-changed=src/kernels");
    println!("cargo:rerun-if-changed=include");

    let mut build = cc::Build::new();
    build.cpp(true);
    build.include("include");

    // Check if CUDA feature is enabled for liminika-core
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // CUDA build configurations will be inserted here
    }

    // Safely compile C/C++ kernel sources only if files are registered
    let has_sources = false;

    // Example:
    // if std::path::Path::new("src/kernels/cpu/example.cpp").exists() {
    //     build.file("src/kernels/cpu/example.cpp");
    //     has_sources = true;
    // }

    if has_sources {
        build.compile("liminika_kernels");
    }
}
