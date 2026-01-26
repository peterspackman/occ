add_library(occ_qm
    "${CMAKE_CURRENT_SOURCE_DIR}/auto_aux_basis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cdiis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/convergence_accelerator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/chelpg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cint_interface.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ediis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/adiis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gradients.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/guess_density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_2c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_3c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_4c.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_esp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_grad1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_grad2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_df.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/merge.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mo_integral_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mp2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/oniom.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orbital_smearing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orthogonalizer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/post_hf_method.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf_method.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/seminumerical_exchange.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial_grid_hierarchy.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/split_ri_j.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/fchkreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/fchkwriter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/gaussian_input_file.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/moldenreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/orca_json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/qcschema.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/wavefunction_json.cpp"
    ${OCC_QM_INCLUDE_FILES}
)

target_link_libraries(occ_qm
    PUBLIC
    occ_core
    occ_gto
    occ_ints
    occ_numint
    occ_disp
    occ_xdm
    nlohmann_json::nlohmann_json
    libcint::libcint
    libecpint_static
)
target_include_directories(occ_qm PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_definitions(occ_qm PUBLIC HAVE_ECPINT=1)

target_compile_features(occ_qm PUBLIC cxx_std_20)

