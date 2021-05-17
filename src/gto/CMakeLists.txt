add_library(occ_gto
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_gto
    PUBLIC 
    occ_core
    occ_qm
)
target_include_directories(occ_gto PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_gto PUBLIC cxx_std_17)

