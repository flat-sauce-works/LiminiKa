# CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(LiminiKa LANGUAGES C CXX)

# Enforce C11 and C++20 specifications for GCSO C-ABI and Kernel execution
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Position Independent Code (PIC) globally for Rust FFI dynamic/static linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to RelWithDebInfo if no build type is specified (Prevents unoptimized/conflicting flags)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()

# Optional GPU backend compilation switches for future kernel integrations
option(LIMINIKA_ENABLE_CUDA "Enable CUDA kernel integration" OFF)
option(LIMINIKA_ENABLE_VULKAN "Enable Vulkan compute kernel integration" OFF)
option(LIMINIKA_ENABLE_METAL "Enable Metal compute kernel integration" OFF)

if(LIMINIKA_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Enforce strict compiler warnings and configuration-aware optimization flags
if(MSVC)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
        $<$<COMPILE_LANGUAGE:C,CXX>:/WX>
        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        $<$<COMPILE_LANGUAGE:C,CXX>:/wd4324> # Suppress structure alignment padding warning C4324
        # Restrict /O2 strictly to Release and RelWithDebInfo configs to eliminate MSVC D8016 (/RTC1 conflict)
        $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>
    )
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
        $<$<COMPILE_LANGUAGE:C,CXX>:-pedantic>
        $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
        # Apply maximum optimization only in release/relwithdebinfo builds
        $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    )
endif()

# Define global header target for LiminiKa C-ABI
add_library(liminika_headers INTERFACE)
target_include_directories(liminika_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Enable CTest and register initial testing target
enable_testing()
add_test(NAME liminika_dummy_test COMMAND ${CMAKE_COMMAND} -E true)

# Include kernels subdirectory if CMakeLists.txt exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/CMakeLists.txt")
    add_subdirectory(src/kernels)
endif()

# Include examples subdirectory if CMakeLists.txt exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
    add_subdirectory(examples)
endif()

# Include tests subdirectory if CMakeLists.txt exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# Configure clang-format target for automatic code formatting in CI/local environments
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        CONFIGURE_DEPENDS
        "include/**/*.h"
        "include/**/*.hpp"
        "include/**/*.inl"
        "include/**/*.c"
        "src/kernels/**/*.cpp"
        "src/kernels/**/*.cc"
        "src/kernels/**/*.cxx"
        "src/kernels/**/*.cu"
        "src/kernels/**/*.metal"
        "src/kernels/**/*.h"
        "src/kernels/**/*.hpp"
        "src/kernels/**/*.inl"
        "src/kernels/**/*.c"
        "examples/**/*.c"
        "examples/**/*.cpp"
        "examples/**/*.cc"
        "examples/**/*.h"
        "examples/**/*.hpp"
        "tests/**/*.cpp"
        "tests/**/*.cc"
        "tests/**/*.c"
        "tests/**/*.h"
        "tests/**/*.hpp"
    )

    if(ALL_CXX_SOURCE_FILES)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_BIN} -i -style=file ${ALL_CXX_SOURCE_FILES}
            COMMENT "Running clang-format on C/C++ source files..."
        )
        add_custom_target(format-check
            COMMAND ${CLANG_FORMAT_BIN} --dry-run --Werror -style=file ${ALL_CXX_SOURCE_FILES}
            COMMENT "Checking C/C++ code formatting with clang-format..."
        )
    else()
        add_custom_target(format
            COMMAND ${CMAKE_COMMAND} -E echo "No C/C++ source files found for formatting."
            COMMENT "Skipping clang-format (no matching source files)..."
        )
        add_custom_target(format-check
            COMMAND ${CMAKE_COMMAND} -E echo "No C/C++ source files found for formatting check."
            COMMENT "Skipping clang-format check (no matching source files)..."
        )
    endif()
endif()