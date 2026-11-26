# name: CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(LiminiKa VERSION 0.4.0 LANGUAGES C CXX)

# Enforce C11 and C++20 specifications for LiminiKa C-ABI compilation
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable Position Independent Code (PIC) globally for Rust FFI static linking
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Default to RelWithDebInfo if no build type is specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose the type of build." FORCE)
endif()

# Optional CPU SIMD vector extension switches (synchronized with C-ABI headers & CI)
option(LIMINIKA_ENABLE_AVX2 "Enable AVX2 SIMD instructions" OFF)
option(LIMINIKA_ENABLE_AVX512 "Enable AVX-512 SIMD instructions" OFF)
option(LIMINIKA_ENABLE_NEON "Enable ARM Neon SIMD instructions" OFF)
option(LIMINIKA_ENABLE_SVE "Enable ARM SVE SIMD instructions" OFF)
option(LIMINIKA_ENABLE_RVV "Enable RISC-V Vector SIMD instructions" OFF)

# Enforce strict compiler warnings, 16MiB main-thread stack size, and SIMD flags
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4 /WX /utf-8 /Zc:preprocessor /Zc:inline /wd4324")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /WX /utf-8 /permissive- /EHsc /Zc:preprocessor /Zc:inline /wd4324")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /O2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:16777216")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|x86_64|x64")
        if(LIMINIKA_ENABLE_AVX512)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX512")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        elseif(LIMINIKA_ENABLE_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        endif()
    endif()
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -Wpedantic -Wconversion -Wno-unused-parameter -Wstrict-prototypes")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wpedantic -Wconversion -Wno-unused-parameter")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -fstrict-aliasing")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fstrict-aliasing")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -O3 -fstrict-aliasing")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -fstrict-aliasing")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|x64")
        if(LIMINIKA_ENABLE_AVX512)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma")
        elseif(LIMINIKA_ENABLE_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2 -mfma")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        endif()
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64" AND LIMINIKA_ENABLE_SVE AND NOT APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8-a+sve")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+sve")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64" AND LIMINIKA_ENABLE_RVV)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gcv")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")
    endif()
endif()

# Define global header target for LiminiKa C-ABI and propagate SIMD preprocessor definitions
if(NOT TARGET liminika_headers)
    add_library(liminika_headers INTERFACE)
    target_include_directories(liminika_headers INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    if(LIMINIKA_ENABLE_AVX2)
        target_compile_definitions(liminika_headers INTERFACE LIMINIKA_ENABLE_AVX2=1)
    endif()
    if(LIMINIKA_ENABLE_AVX512)
        target_compile_definitions(liminika_headers INTERFACE LIMINIKA_ENABLE_AVX512=1)
    endif()
    if(LIMINIKA_ENABLE_NEON)
        target_compile_definitions(liminika_headers INTERFACE LIMINIKA_ENABLE_NEON=1)
    endif()
    if(LIMINIKA_ENABLE_SVE)
        target_compile_definitions(liminika_headers INTERFACE LIMINIKA_ENABLE_SVE=1)
    endif()
    if(LIMINIKA_ENABLE_RVV)
        target_compile_definitions(liminika_headers INTERFACE LIMINIKA_ENABLE_RVV=1)
    endif()
endif()

# Enable CTest (tests are registered in examples/c_api/CMakeLists.txt and tests/CMakeLists.txt)
enable_testing()

# Locate the Rust-built liminika_core static library (staticlib preferred so CTest runs standalone)
find_library(LIMINIKA_CORE_LIB
    NAMES libliminika_core.a liminika_core.lib liminika_core
    PATHS "${CMAKE_SOURCE_DIR}/target/release"
          "${CMAKE_SOURCE_DIR}/target/debug"
    NO_DEFAULT_PATH
)

if(NOT LIMINIKA_CORE_LIB)
    message(FATAL_ERROR
        "Rust static library `liminika_core` was not found in "
        "`${CMAKE_SOURCE_DIR}/target/release` or `${CMAKE_SOURCE_DIR}/target/debug`. "
        "Run `cargo build --release -p liminika-core` before configuring CMake, "
        "or pass `-DLIMINIKA_CORE_LIB=/path/to/libliminika_core.a`."
    )
endif()

add_library(liminika_core_ffi INTERFACE)
target_link_libraries(liminika_core_ffi INTERFACE
    liminika_headers
    "${LIMINIKA_CORE_LIB}"
)
if(WIN32)
    target_link_libraries(liminika_core_ffi INTERFACE
        ws2_32
        userenv
        bcrypt
        ntdll
        advapi32
        synchronization
        dbghelp
    )
elseif(UNIX)
    target_link_libraries(liminika_core_ffi INTERFACE
        m
        pthread
        dl
    )
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
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/c_api/*.c"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/c_api/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/c_api/*.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/c_api/*.h"
        "${CMAKE_CURRENT_SOURCE_DIR}/examples/c_api/*.hpp"
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
