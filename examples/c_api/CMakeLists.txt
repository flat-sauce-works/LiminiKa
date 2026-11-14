# name: examples/c_api/CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

set(LIMINIKA_CORE_LIB_DIR "${CMAKE_SOURCE_DIR}/target/release" CACHE PATH "Directory containing compiled liminika_core library")
find_library(LIMINIKA_CORE_LIB
    NAMES liminika_core liminika_core.dll
    HINTS "${LIMINIKA_CORE_LIB_DIR}" "${CMAKE_SOURCE_DIR}/target/debug"
)

if(LIMINIKA_CORE_LIB)
    add_executable(gcso_phi4_gguf_hook_example phi4_gguf_hook_example.c)
    target_link_libraries(gcso_phi4_gguf_hook_example PRIVATE liminika_headers "${LIMINIKA_CORE_LIB}")
    if(WIN32)
        target_link_libraries(gcso_phi4_gguf_hook_example PRIVATE ws2_32 userenv bcrypt ntdll)
    else()
        target_link_libraries(gcso_phi4_gguf_hook_example PRIVATE m pthread dl)
    endif()
    add_test(NAME gcso_phi4_gguf_hook_example_test COMMAND gcso_phi4_gguf_hook_example)
else()
    message(STATUS "liminika_core library not found yet; skipping gcso_phi4_gguf_hook_example (run cargo build -p liminika-core first).")
endif()
