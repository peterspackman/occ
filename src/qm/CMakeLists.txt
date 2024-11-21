add_library(occ_qm
    "${CMAKE_CURRENT_SOURCE_DIR}/cdiis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/chelpg.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cint_interface.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ediis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/guess_density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_grad1.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integral_engine_df.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/merge.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mo.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/oniom.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/partitioning.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/orbital_smearing.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shell.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf_method.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction.cpp"
    ${OCC_QM_INCLUDE_FILES}
)

target_link_libraries(occ_qm
    PUBLIC 
    occ_core
    occ_io
    occ_gto
    libcint::libcint
    libecpint_static
)
target_include_directories(occ_qm PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_definitions(occ_qm PUBLIC HAVE_ECPINT=1)

target_compile_features(occ_qm PUBLIC cxx_std_20)

