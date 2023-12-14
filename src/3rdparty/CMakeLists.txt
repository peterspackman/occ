add_library(_subprocess
    STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/CowData.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/environ.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pipe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/PipeVar.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProcessBuilder.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProcessBuilder_posix.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ProcessBuilder_windows.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shell_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utf8_to_utf16.cpp"
)

target_link_libraries(_subprocess
    PUBLIC 
    Threads::Threads
)

target_compile_features(_subprocess PUBLIC cxx_std_20)

set(OCC_3RDPARTY_SUBPROCESS_INCLUDE_DIR
    "${OCC_INCLUDE_DIR}/occ/3rdparty/subprocess/"
)

target_include_directories(_subprocess 
    PUBLIC 
    ${OCC_3RDPARTY_SUBPROCESS_INCLUDE_DIR}
)

if("${MSVC}")
    target_compile_options(_subprocess PUBLIC -Zc:__cplusplus)
endif()

add_subdirectory(gau2grid)
add_subdirectory(libecpint)
