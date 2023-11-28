add_library(occ_interaction
    "${CMAKE_CURRENT_SOURCE_DIR}/coulomb.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/disp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pairinteraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_potential.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/polarization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wolf.cpp"
    ${OCC_INTERACTION_INCLUDE_FILES}
)

target_link_libraries(occ_interaction
    PUBLIC 
    occ_core
    occ_crystal
    occ_qm
    occ_io
    occ_xdm
)
target_include_directories(occ_interaction PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_interaction PUBLIC cxx_std_20)
