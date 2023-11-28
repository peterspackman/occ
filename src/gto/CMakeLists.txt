add_library(occ_gto
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rotation.cpp"
    ${OCC_GTO_INCLUDE_FILES}
)

target_link_libraries(occ_gto
    PUBLIC 
    occ_core
    occ_qm
    gau2grid_static
)
target_include_directories(occ_gto PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_gto PUBLIC cxx_std_20)

