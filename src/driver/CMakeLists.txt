add_library(occ_driver
    "${CMAKE_CURRENT_SOURCE_DIR}/geometry_optimization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_growth.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_point.cpp"
    ${OCC_DRIVER_INCLUDE_FILES}
)

target_link_libraries(occ_driver
    PUBLIC 
    occ_io
    occ_interaction
    occ_xtb
    occ_dft
    occ_qm
    occ_solvent
    occ_crystal
    occ_cg
)
target_include_directories(occ_driver PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_driver PUBLIC cxx_std_20)
