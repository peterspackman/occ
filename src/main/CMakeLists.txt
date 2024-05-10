add_library(occ_main
    "${CMAKE_CURRENT_SOURCE_DIR}/cli_validators.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_surface_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/geometry_optimization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/monomer_wavefunctions.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/point_functors.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/properties.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_point.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_isosurface.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_cg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_cube.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_describe.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_elat.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_pair.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_dimers.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solvation_partition.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp"
    ${OCC_MAIN_INCLUDE_FILES}
)

target_link_libraries(occ_main
    PUBLIC 
    occ_core
    occ_xtb
    occ_qm
    occ_dft
    occ_solvent
    occ_interaction
    occ_isosurface
    occ_crystal
    occ_slater
    occ_geometry
    occ_gto
    occ_sht
    occ_descriptors
    CLI11::CLI11
)
target_include_directories(occ_main PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_main PUBLIC cxx_std_20)

add_executable(xdm EXCLUDE_FROM_ALL 
    "${CMAKE_CURRENT_SOURCE_DIR}/xdm.cpp")
target_link_libraries(xdm
    PUBLIC
    occ_main
    occ_xdm
)

set_target_properties(xdm
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(occ "${CMAKE_CURRENT_SOURCE_DIR}/occ.cpp")
target_link_libraries(occ
    PUBLIC
    occ_main
)
set_target_properties(occ
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Use for GCC time report when compiling occ
# target_compile_options(occ PRIVATE "-ftime-report")

add_executable(make_radial_interpolators EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_SOURCE_DIR}/make_radial_interpolators.cpp")
target_link_libraries(make_radial_interpolators PUBLIC occ_slater)
set_target_properties(make_radial_interpolators
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(occ-mlx EXCLUDE_FROM_ALL 
    "${CMAKE_CURRENT_SOURCE_DIR}/mlx.cpp")
target_link_libraries(occ-mlx
    PUBLIC
    mlx
    Eigen3::Eigen
)
target_compile_features(occ-mlx PUBLIC cxx_std_20)

add_executable(make_atomic_pair_potentials EXCLUDE_FROM_ALL 
    "${CMAKE_CURRENT_SOURCE_DIR}/make_atomic_pair_potentials.cpp")
target_link_libraries(make_atomic_pair_potentials 
    PUBLIC
    occ_main
)
target_compile_features(make_atomic_pair_potentials PUBLIC cxx_std_20)
set_target_properties(make_atomic_pair_potentials 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
