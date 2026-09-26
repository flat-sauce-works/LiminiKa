cmake_minimum_required(VERSION 3.18)
project(LiminiKa LANGUAGES C CXX)

# Set C++ standard specifications
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include directories for project headers
include_directories(include)

# Enable CTest and register initial testing target
enable_testing()
add_test(NAME liminika_dummy_test COMMAND ${CMAKE_COMMAND} -E true)

# Configure clang-format target for automatic code formatting in CI/local environments
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    # Recursively collect C/C++ header and source files across subdirectories
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        CONFIGURE_DEPENDS
        "include/*.h"
        "include/*.hpp"
        "src/kernels/*.cpp"
        "src/kernels/*.cu"
        "src/kernels/*.metal"
        "src/kernels/*.h"
        "src/kernels/*.hpp"
    )

    if(ALL_CXX_SOURCE_FILES)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_BIN} -i -style=file ${ALL_CXX_SOURCE_FILES}
            COMMENT "Running clang-format on C/C++ source files..."
        )
    else()
        # Fallback dummy format target when no source files match the glob pattern
        add_custom_target(format
            COMMAND ${CMAKE_COMMAND} -E echo "No C/C++ source files found for formatting."
            COMMENT "Skipping clang-format (no matching source files)..."
        )
    endif()
endif()

# Dummy interface target to keep CMake build pipeline clean during initial setup
add_library(liminika_dummy_kernels INTERFACE)