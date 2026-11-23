# name: examples/CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/c_api/CMakeLists.txt")
    add_subdirectory(c_api)
endif()
