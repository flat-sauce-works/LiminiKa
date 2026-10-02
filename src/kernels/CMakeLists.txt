# SPDX-License-Identifier: MIT OR Apache-2.0

# -------------------------------------------------------------------
# LiminiKa GCSO Kernels Build Configuration
# Targets: C11 / C++20 CPU, CUDA, Metal, and Vulkan Compute Kernels
# -------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18)
project(liminika_kernels LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Option definitions for target hardware backends
option(LIMINIKA_ENABLE_CUDA "Enable CUDA compute backend" OFF)
option(LIMINIKA_ENABLE_VULKAN "Enable Vulkan compute backend" OFF)
option(LIMINIKA_ENABLE_METAL "Enable Metal compute backend" OFF)

# Header interface target for C-ABI layout definitions
if(NOT TARGET liminika_headers)
    add_library(liminika_headers INTERFACE)
    target_include_directories(liminika_headers INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
        $<INSTALL_INTERFACE:include>
    )
endif()

# List of static targets to install for Cargo output consumption
set(INSTALL_TARGET_LIST "")

# Collect common and CPU kernel sources safely
file(GLOB_RECURSE KERNEL_COMMON_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.c"
)

file(GLOB_RECURSE KERNEL_CPU_EXTRA_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cpu/*.c"
)

set(KERNEL_CPU_SOURCES ${KERNEL_COMMON_SOURCES} ${KERNEL_CPU_EXTRA_SOURCES})

# Fallback dummy symbol generation if source files are missing in early setups
if(NOT KERNEL_CPU_SOURCES)
    set(KERNEL_CPU_SOURCES "${CMAKE_CURRENT_BINARY_DIR}/dummy_kernel.cpp")
    file(WRITE "${KERNEL_CPU_SOURCES}" "// SPDX-License-Identifier: MIT OR Apache-2.0\nextern \"C\" void gcso_dummy_kernel_symbol(void) {}\n")
endif()

add_library(liminika_kernels_cpu STATIC ${KERNEL_CPU_SOURCES})
target_link_libraries(liminika_kernels_cpu PUBLIC liminika_headers)
target_compile_features(liminika_kernels_cpu PUBLIC c_std_11 cxx_std_20)
set_target_properties(liminika_kernels_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Scoped compile definitions for conditional ABI compilation
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
    target_compile_options(liminika_kernels_cpu PRIVATE /W4 /O2 /permissive- /utf-8)
else()
    target_compile_options(liminika_kernels_cpu PRIVATE -Wall -Wextra -O3 -fstrict-aliasing)
endif()

list(APPEND INSTALL_TARGET_LIST liminika_kernels_cpu)

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
            list(APPEND INSTALL_TARGET_LIST liminika_kernels_cuda)
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
        list(APPEND INSTALL_TARGET_LIST liminika_kernels_vulkan)
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
        list(APPEND INSTALL_TARGET_LIST liminika_kernels_metal)
    endif()
endif()

# Mandatory CMake install rules required by Cargo cmake build script
install(TARGETS ${INSTALL_TARGET_LIST}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../../include/liminika"
    DESTINATION include
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)