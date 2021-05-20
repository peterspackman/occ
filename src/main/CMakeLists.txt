add_executable(occ "${CMAKE_CURRENT_SOURCE_DIR}/occ.cpp")
target_link_libraries(occ
    PUBLIC
    occ_qm
    occ_dft
)
set_target_properties(occ
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(occ-pair "${CMAKE_CURRENT_SOURCE_DIR}/occ_pair.cpp")
target_link_libraries(occ-pair PUBLIC occ_interaction)
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

add_executable(cosmo "${CMAKE_CURRENT_SOURCE_DIR}/cosmo.cpp")
target_link_libraries(cosmo
    PUBLIC
    occ_qm
    occ_dft
    occ_geometry
    occ_slater
    occ_solvent
)
set_target_properties(cosmo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


add_executable(crystal_interactions "${CMAKE_CURRENT_SOURCE_DIR}/interactions.cpp")
target_link_libraries(crystal_interactions
    PUBLIC
    occ_qm
    occ_dft
    occ_solvent
    occ_io
    occ_crystal
    occ_interaction
)

set_target_properties(crystal_interactions
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(hs "${CMAKE_CURRENT_SOURCE_DIR}/hs.cpp")
target_link_libraries(hs
    PUBLIC
    occ_geometry
    occ_slater
    occ_solvent
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
)

set_target_properties(occ-elat
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)