add_library(occ_interaction
    "${CMAKE_CURRENT_SOURCE_DIR}/ce_energy_model.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/coulomb.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/disp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interaction_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lattice_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_energy_store.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pairinteraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_potential.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/polarization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction_transform.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wolf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xtb_energy_model.cpp"
    ${OCC_INTERACTION_INCLUDE_FILES}
)

target_link_libraries(occ_interaction
    PUBLIC 
    occ_core
    occ_crystal
    occ_qm
    occ_io
    occ_xdm
    occ_xtb
)
target_include_directories(occ_interaction PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_interaction PUBLIC cxx_std_20)
