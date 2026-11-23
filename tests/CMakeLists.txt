# name: tests/CMakeLists.txt
# SPDX-License-Identifier: MIT OR Apache-2.0

add_executable(liminika_c_abi_smoke_test
    "${CMAKE_CURRENT_SOURCE_DIR}/ffi/test_c_abi_smoke.cpp"
)
target_link_libraries(liminika_c_abi_smoke_test PRIVATE liminika_core_ffi)
target_compile_features(liminika_c_abi_smoke_test PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(liminika_c_abi_smoke_test PRIVATE /W4 /EHsc /wd4324 /permissive- /utf-8 /Zc:preprocessor)
else()
    target_compile_options(liminika_c_abi_smoke_test PRIVATE -Wall -Wextra -Werror)
endif()
add_test(NAME liminika_c_abi_smoke_test COMMAND liminika_c_abi_smoke_test)
