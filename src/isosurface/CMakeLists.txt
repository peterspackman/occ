add_library(occ_isosurface
    "${CMAKE_CURRENT_SOURCE_DIR}/curvature.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/deformation_density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/eeq_esp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/electric_potential.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/electron_density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/isosurface.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orbital_index.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/promolecule_density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/stockholder_weight.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/surface_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/void.cpp"
    ${OCC_ISOSURFACE_INCLUDE_FILES}
)

target_link_libraries(occ_isosurface
    PUBLIC 
    occ_core
    occ_qm
    occ_dft
    occ_crystal
    occ_slater
    occ_geometry
    occ_gto
    Threads::Threads
    unordered_dense::unordered_dense
)
target_include_directories(occ_isosurface PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_isosurface PUBLIC cxx_std_20)

