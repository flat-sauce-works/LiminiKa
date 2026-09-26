cmake_minimum_required(VERSION 3.18)
project(LiminiKa LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Configure clang-format target
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        include/*.h src/kernels/*.cu src/kernels/*.metal src/kernels/*.cpp
    )
    if(ALL_CXX_SOURCE_FILES)
        add_custom_target(format
            COMMAND ${CLANG_FORMAT_BIN} -i -style=file ${ALL_CXX_SOURCE_FILES}
            COMMENT "Running clang-format on C/C++ source files..."
        )
    endif()
endif()

# Dummy interface target to keep CMake build steps clean during initial setup
add_library(liminika_dummy_kernels INTERFACE)