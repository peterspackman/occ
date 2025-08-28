add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(main)

if("${WITH_PYTHON_BINDINGS}")

set(PYTHON_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/trajanpy.cpp"
)

nanobind_add_module(_trajanpy
    NB_STATIC
    LTO
    STABLE_ABI
    ${PYTHON_BINDING_SOURCES}
)

target_include_directories(_trajanpy PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/python"
)

target_compile_definitions(_trajanpy PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(_trajanpy PUBLIC cxx_std_20)
target_link_libraries(_trajanpy PUBLIC 
    trajan_main
)

install(TARGETS _trajanpy LIBRARY DESTINATION trajanpy)
endif()
