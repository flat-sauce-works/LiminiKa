# SPDX-License-Identifier: MIT OR Apache-2.0

# -------------------------------------------------------------------
# LiminiKa GCSO Kernels Build Configuration
# Targets: C11 / C++20 CPU, CUDA, Metal, and Vulkan Compute Kernels
# -------------------------------------------------------------------

# Provide an interface target for subprojects and Rust build.rs linkage
add_library(liminika_kernels INTERFACE)
target_link_libraries(liminika_kernels INTERFACE liminika_headers)

# Collect C/C++ source files across CPU and common kernel subdirectories
file(GLOB_RECURSE KERNEL_CPU_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c"
)

# Build CPU static library if kernel source files exist
if(KERNEL_CPU_SOURCES)
    add_library(liminika_kernels_cpu STATIC ${KERNEL_CPU_SOURCES})
    target_link_libraries(liminika_kernels_cpu PUBLIC liminika_headers)
    target_compile_features(liminika_kernels_cpu PUBLIC c_std_11 cxx_std_20)
    set_target_properties(liminika_kernels_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(liminika_kernels INTERFACE liminika_kernels_cpu)
endif()

# Optional CUDA compute backend integration
if(LIMINIKA_ENABLE_CUDA AND CMAKE_CUDA_COMPILER_LOADED)
    file(GLOB_RECURSE KERNEL_CUDA_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu"
    )
    if(KERNEL_CUDA_SOURCES)
        add_library(liminika_kernels_cuda STATIC ${KERNEL_CUDA_SOURCES})
        target_link_libraries(liminika_kernels_cuda PUBLIC liminika_headers)
        set_target_properties(liminika_kernels_cuda PROPERTIES
            CUDA_STANDARD 20
            POSITION_INDEPENDENT_CODE ON
        )
        target_link_libraries(liminika_kernels INTERFACE liminika_kernels_cuda)
    endif()
endif()