add_executable(tonto-ng "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tonto-ng
    PUBLIC
    tonto_qm
    tonto_dft
)
set_target_properties(tonto-ng
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(tonto-ng-ce "${CMAKE_CURRENT_SOURCE_DIR}/ce_main.cpp")
target_link_libraries(tonto-ng-ce PUBLIC tonto_interaction)
set_target_properties(tonto-ng-ce
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(make_radial_interpolators "${CMAKE_CURRENT_SOURCE_DIR}/make_radial_interpolators.cpp")
target_link_libraries(make_radial_interpolators PUBLIC tonto_slater)
set_target_properties(make_radial_interpolators
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(cosmo "${CMAKE_CURRENT_SOURCE_DIR}/cosmo.cpp")
target_link_libraries(cosmo
    PUBLIC
    tonto_qm
    tonto_dft
    tonto_geometry
    tonto_slater
    tonto_solvent
)
set_target_properties(cosmo
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
