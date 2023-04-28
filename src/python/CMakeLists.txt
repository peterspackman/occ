nanobind_add_module(_occ 
    NB_STATIC
    LTO
    "${CMAKE_CURRENT_SOURCE_DIR}/module.cpp")
target_compile_definitions(_occ PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_link_libraries(_occ PUBLIC occ_main)

install(TARGETS _occ DESTINATION .)
