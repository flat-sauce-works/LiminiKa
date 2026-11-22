# name: examples/c_api/CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

add_executable(gcso_phi4_gguf_hook_example phi4_gguf_hook_example.c)
target_link_libraries(gcso_phi4_gguf_hook_example PRIVATE liminika_core_ffi)
target_compile_features(gcso_phi4_gguf_hook_example PRIVATE c_std_11)
add_test(NAME gcso_phi4_gguf_hook_example_test COMMAND gcso_phi4_gguf_hook_example)
