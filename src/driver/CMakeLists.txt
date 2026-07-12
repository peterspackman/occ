add_library(occ_driver_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/geometry_optimization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_growth.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cg_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_surface_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_morphology.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/monomer_wavefunctions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_point.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dma_driver.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/vibrational_analysis.cpp"
    ${OCC_DRIVER_INCLUDE_FILES}
)
add_library(occ_driver $<TARGET_OBJECTS:occ_driver_obj>)
target_include_directories(occ_driver_obj PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR})
target_link_libraries(
    occ_driver_obj
    PUBLIC
        occ_io
        occ_interaction
        occ_xtb
        occ_dft
        occ_qm
        occ_xdm
        occ_correlation
        occ_cc
        occ_dma
        occ_solvent
        occ_crystal
        occ_cg
        occ_geometry
        occ_isosurface
        occ_opt
)

target_link_libraries(
    occ_driver
    PUBLIC
        occ_io
        occ_interaction
        occ_xtb
        occ_dft
        occ_qm
        occ_xdm
        occ_correlation
        occ_cc
        occ_dma
        occ_solvent
        occ_crystal
        occ_cg
        occ_geometry
        occ_isosurface
        occ_opt
)
target_include_directories(
    occ_driver
    PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_driver PUBLIC cxx_std_20)
