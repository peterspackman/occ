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
    cartesian_interaction.cpp
    cartesian_molecule.cpp
    cartesian_force.cpp
    multipole_source.cpp
    lbfgs.cpp
    trust_region.cpp
    crystal_energy.cpp
    crystal_optimizer.cpp
    cartesian_hessian.cpp
    dmacrys_input.cpp
    crystal_strain.cpp
    ewald_sum.cpp
)

target_include_directories(occ_mults PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(occ_mults PUBLIC
    occ_core
    occ_dma
    occ_sht
    occ_crystal
    nlohmann_json::nlohmann_json
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

