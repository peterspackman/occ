add_library(occ_numint
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lebedev.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid_points.cpp"
    ${OCC_NUMINT_INCLUDE_FILES}
)

target_link_libraries(occ_numint
    PUBLIC
    occ_core
    occ_gto
    occ_io
)
target_include_directories(occ_numint PUBLIC
    ${OCC_INCLUDE_DIR}
)

target_compile_features(occ_numint PUBLIC cxx_std_20)
