add_library(occ_mults
    rotation.cpp
    sfunctions.cpp
    sfunctions_s1.cpp
    sfunctions_s2.cpp
    sfunctions_s3.cpp
    sfunctions_s3_kernels.cpp
    sfunctions_s4.cpp
    sfunctions_s4_kernels.cpp
    sfunctions_s5.cpp
    sfunctions_s5_kernels.cpp
    esp.cpp
    orient_io.cpp
    sfunction_evaluator.cpp
    sfunction_term_builder.cpp
    multipole_interactions.cpp
    derivative_transform.cpp
    short_range.cpp
    torque.cpp
    rigid_body.cpp
    rigid_body_dynamics.cpp
    optimization_projection.cpp
)

target_include_directories(occ_mults PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(occ_mults PUBLIC 
    occ_core
    occ_dma
    occ_sht
)

target_compile_features(occ_mults PUBLIC cxx_std_20)

set_target_properties(occ_mults PROPERTIES
    EXPORT_NAME mults
)

install(TARGETS occ_mults
    EXPORT occTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Multipole interaction command-line program
add_executable(interaction_main interaction_main.cpp)
target_link_libraries(interaction_main
    occ_mults
    occ_core
    occ_dma
    nlohmann_json::nlohmann_json
)
target_compile_features(interaction_main PUBLIC cxx_std_20)

install(TARGETS interaction_main
    RUNTIME DESTINATION bin
)
