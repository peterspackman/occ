add_library(occ_xdm
    "${CMAKE_CURRENT_SOURCE_DIR}/becke_hole.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xdm.cpp"
    ${OCC_XDM_INCLUDE_FILES}
)

target_link_libraries(occ_xdm
    PUBLIC 
    occ_qm
    occ_dft
    occ_slater
)
target_include_directories(occ_xdm PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_xdm PUBLIC cxx_std_20)

