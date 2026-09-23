# CMakeLists.txt
find_program(CLANG_FORMAT_BIN clang-format)
if(CLANG_FORMAT_BIN)
    file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
        include/*.h src/core/*.cpp src/core/*.h src/kernels/*.cu src/kernels/*.metal
    )
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_BIN} -i -style=file ${ALL_CXX_SOURCE_FILES}
        COMMENT "Running clang-format..."
    )
endif()