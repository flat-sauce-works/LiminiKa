# name: src/kernels/CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

# -------------------------------------------------------------------
# LiminiKa GCSO Kernels Build Configuration
# Targets: C11 / C++20 CPU, CUDA, Metal, and Vulkan Compute Kernels
# -------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)

# Header interface target for C-ABI layout definitions
if(NOT TARGET liminika_headers)
    add_library(liminika_headers INTERFACE)
    target_include_directories(liminika_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
    )
endif()

# Interface target linking kernel libraries to external subprojects
if(NOT TARGET liminika_kernels)
    add_library(liminika_kernels INTERFACE)
    target_link_libraries(liminika_kernels INTERFACE liminika_headers)
endif()

# Explicit source listing for common GCSO domain modules
set(KERNEL_COMMON_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/common/system.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/runtime_context.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/spt_action_hub.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/dpsr_kernel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/attractor_edbc.cpp"
)

# Collect platform-specific CPU kernel sources if present
file(GLOB_RECURSE KERNEL_CPU_EXTRA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c"
)

set(KERNEL_CPU_SOURCES ${KERNEL_COMMON_SOURCES} ${KERNEL_CPU_EXTRA_SOURCES})

# Build CPU static library
if(KERNEL_CPU_SOURCES)
    add_library(liminika_kernels_cpu STATIC ${KERNEL_CPU_SOURCES})
    target_link_libraries(liminika_kernels_cpu PUBLIC liminika_headers)
    target_compile_features(liminika_kernels_cpu PUBLIC c_std_11 cxx_std_20)
    set_target_properties(liminika_kernels_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Scoped feature definitions for system capability queries
    if(LIMINIKA_ENABLE_CUDA)
        target_compile_definitions(liminika_kernels_cpu PRIVATE LIMINIKA_ENABLE_CUDA)
    endif()
    if(LIMINIKA_ENABLE_VULKAN)
        target_compile_definitions(liminika_kernels_cpu PRIVATE LIMINIKA_ENABLE_VULKAN)
    endif()
    if(LIMINIKA_ENABLE_METAL)
        target_compile_definitions(liminika_kernels_cpu PRIVATE LIMINIKA_ENABLE_METAL)
    endif()

    if(MSVC)
        target_compile_options(liminika_kernels_cpu PRIVATE /W4 /WX /O2 /permissive- /utf-8)
    else()
        target_compile_options(liminika_kernels_cpu PRIVATE -Wall -Wextra -Werror -O3 -fstrict-aliasing)
    endif()

    target_link_libraries(liminika_kernels INTERFACE liminika_kernels_cpu)
endif()

# Optional CUDA compute backend integration
if(LIMINIKA_ENABLE_CUDA)
    enable_language(CUDA OPTIONAL)
    if(CMAKE_CUDA_COMPILER_LOADED)
        file(GLOB_RECURSE KERNEL_CUDA_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/cuda/*.cu"
        )
        if(KERNEL_CUDA_SOURCES)
            add_library(liminika_kernels_cuda STATIC ${KERNEL_CUDA_SOURCES})
            target_link_libraries(liminika_kernels_cuda PUBLIC liminika_headers)
            target_compile_definitions(liminika_kernels_cuda PRIVATE LIMINIKA_ENABLE_CUDA)
            set_target_properties(liminika_kernels_cuda PROPERTIES
                CUDA_STANDARD 20
                POSITION_INDEPENDENT_CODE ON
            )
            target_link_libraries(liminika_kernels INTERFACE liminika_kernels_cuda)
        endif()
    endif()
endif()

# Optional Vulkan compute backend integration
if(LIMINIKA_ENABLE_VULKAN)
    file(GLOB_RECURSE KERNEL_VULKAN_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/vulkan/*.c"
    )
    if(KERNEL_VULKAN_SOURCES)
        add_library(liminika_kernels_vulkan STATIC ${KERNEL_VULKAN_SOURCES})
        target_link_libraries(liminika_kernels_vulkan PUBLIC liminika_headers)
        target_compile_definitions(liminika_kernels_vulkan PRIVATE LIMINIKA_ENABLE_VULKAN)
        target_compile_features(liminika_kernels_vulkan PUBLIC c_std_11 cxx_std_20)
        set_target_properties(liminika_kernels_vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(liminika_kernels INTERFACE liminika_kernels_vulkan)
    endif()
endif()

# Optional Metal compute backend integration
if(LIMINIKA_ENABLE_METAL)
    file(GLOB_RECURSE KERNEL_METAL_SOURCES
        CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/metal/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/metal/*.mm"
    )
    if(KERNEL_METAL_SOURCES)
        add_library(liminika_kernels_metal STATIC ${KERNEL_METAL_SOURCES})
        target_link_libraries(liminika_kernels_metal PUBLIC liminika_headers)
        target_compile_definitions(liminika_kernels_metal PRIVATE LIMINIKA_ENABLE_METAL)
        target_compile_features(liminika_kernels_metal PUBLIC cxx_std_20)
        set_target_properties(liminika_kernels_metal PROPERTIES POSITION_INDEPENDENT_CODE ON)
        target_link_libraries(liminika_kernels INTERFACE liminika_kernels_metal)
    endif()
endif()