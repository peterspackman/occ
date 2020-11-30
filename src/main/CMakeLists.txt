add_executable(tontox "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tontox
    PUBLIC
    tonto_qm
)
set_target_properties(tontox
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(ce "${CMAKE_CURRENT_SOURCE_DIR}/ce_main.cpp")
target_link_libraries(ce PUBLIC tonto_interaction)
set_target_properties(ce
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(make_radial_interpolators "${CMAKE_CURRENT_SOURCE_DIR}/make_radial_interpolators.cpp")
target_link_libraries(make_radial_interpolators PUBLIC tonto_slater)
set_target_properties(ce
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
