# name: CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(LiminiKa VERSION 0.1.1 LANGUAGES C CXX)

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

# Optional GPU backend compilation switches
option(LIMINIKA_ENABLE_CUDA "Enable CUDA kernel integration" OFF)
option(LIMINIKA_ENABLE_VULKAN "Enable Vulkan compute kernel integration" OFF)
option(LIMINIKA_ENABLE_METAL "Enable Metal compute kernel integration" OFF)

# Optional CPU SIMD vector extension switches (synchronized with src/kernels/CMakeLists.txt & CI)
option(LIMINIKA_ENABLE_AVX2 "Enable AVX2 SIMD instructions" OFF)
option(LIMINIKA_ENABLE_AVX512 "Enable AVX-512 SIMD instructions" OFF)
option(LIMINIKA_ENABLE_NEON "Enable ARM Neon SIMD instructions" OFF)
option(LIMINIKA_ENABLE_SVE "Enable ARM SVE SIMD instructions" OFF)
option(LIMINIKA_ENABLE_RVV "Enable RISC-V Vector SIMD instructions" OFF)

if(LIMINIKA_ENABLE_METAL AND NOT APPLE)
  message(STATUS "Metal backend is restricted to Apple platforms. Disabling LIMINIKA_ENABLE_METAL.")
  set(LIMINIKA_ENABLE_METAL OFF CACHE BOOL "Enable Metal compute kernel integration" FORCE)
endif()

if(LIMINIKA_ENABLE_CUDA)
  enable_language(CUDA OPTIONAL)
  if(CMAKE_CUDA_COMPILER_LOADED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
  endif()
endif()

# Enforce strict compiler warnings and configuration-aware optimization flags
if(MSVC)
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
    $<$<COMPILE_LANGUAGE:C,CXX>:/WX>
    $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
    $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:C,CXX>:/Zc:inline>
    $<$<COMPILE_LANGUAGE:C,CXX>:/wd4324>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>
  )
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:C>:-Wstrict-prototypes>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>
    $<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-fstrict-aliasing>
  )
endif()

# Define global header target for LiminiKa C-ABI
if(NOT TARGET liminika_headers)
  add_library(liminika_headers INTERFACE)
  target_include_directories(liminika_headers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
endif()

# Enable CTest and register baseline C-ABI test target
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
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cxx"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.mm"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.inl"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
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
