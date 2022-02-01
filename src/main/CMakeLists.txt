add_library(occ_main
    "${CMAKE_CURRENT_SOURCE_DIR}/pair_energy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/properties.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/single_point.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_main
    PUBLIC 
    occ_core
    occ_qm
    occ_dft
    occ_solvent
)
target_include_directories(occ_main PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_main PUBLIC cxx_std_17)


add_executable(occ "${CMAKE_CURRENT_SOURCE_DIR}/occ.cpp")
target_link_libraries(occ
    PUBLIC
    occ_main
    cxxopts
)
set_target_properties(occ
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Use for GCC time report when compiling occ
# target_compile_options(occ PRIVATE "-ftime-report")

add_executable(occ-pair "${CMAKE_CURRENT_SOURCE_DIR}/occ_pair.cpp")
target_link_libraries(occ-pair PUBLIC occ_interaction cxxopts)
set_target_properties(occ-pair
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(make_radial_interpolators "${CMAKE_CURRENT_SOURCE_DIR}/make_radial_interpolators.cpp")
target_link_libraries(make_radial_interpolators PUBLIC occ_slater)
set_target_properties(make_radial_interpolators
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(occ-cg "${CMAKE_CURRENT_SOURCE_DIR}/occ_cg.cpp")
target_link_libraries(occ-cg
    PUBLIC
    occ_qm
    occ_dft
    occ_solvent
    occ_io
    occ_crystal
    occ_interaction
    occ_main
    cxxopts
)

set_target_properties(occ-cg
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(hs "${CMAKE_CURRENT_SOURCE_DIR}/hs.cpp")
target_link_libraries(hs
    PUBLIC
    occ_geometry
    occ_slater
    occ_solvent
    occ_io
    cxxopts
)
set_target_properties(hs
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(occ-elat "${CMAKE_CURRENT_SOURCE_DIR}/occ_elat.cpp")
target_link_libraries(occ-elat
    PUBLIC
    occ_qm
    occ_dft
    occ_io
    occ_crystal
    occ_interaction
    occ_main
    cxxopts
)

set_target_properties(occ-elat
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
