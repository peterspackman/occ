add_library(occ_io
    "${CMAKE_CURRENT_SOURCE_DIR}/basis_g94.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystalgrower.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cube.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dftb_gen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/engrad.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkwriter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gaussian_input_file.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gmf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/json_basis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmcpp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/load_geometry.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/moldenreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/obj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_input.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orca_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ply.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/qcschema.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvent_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tinyply.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xyz.cpp"
    ${OCC_IO_INCLUDE_FILES}
)

target_link_libraries(occ_io
    PUBLIC 
    occ_core
    occ_crystal
    occ_solvent
    occ_qm
    gemmi::gemmi
    nlohmann_json::nlohmann_json
)
target_include_directories(occ_io PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_io PUBLIC cxx_std_20)
