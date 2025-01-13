add_library(occ_cg
    "${CMAKE_CURRENT_SOURCE_DIR}/distance_partition.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interaction_mapper.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/neighbor_atoms.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/result_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/smd_solvation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvation_contribution.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvation_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvent_surface.cpp"
    ${OCC_CG_INCLUDE_FILES}
)

target_link_libraries(occ_cg
    PUBLIC 
    occ_io
    occ_xtb
    occ_dft
    occ_solvent
    occ_interaction
    occ_crystal
)
target_include_directories(occ_cg PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_cg PUBLIC cxx_std_20)
