add_library(occ_io
    "${CMAKE_CURRENT_SOURCE_DIR}/basis_g94.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cifwriter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystalgrower.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cube.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dftb_gen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/engrad.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gmf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/periodic_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_settings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/isosurface_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kmcpp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/load_geometry.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/obj.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_input.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ply.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shelxfile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvent_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tinyply.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xyz.cpp"
    ${OCC_IO_INCLUDE_FILES}
)

target_link_libraries(occ_io
    PUBLIC 
    occ_core
    occ_crystal
    occ_isosurface
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
