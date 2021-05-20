add_library(occ_qm
    "${CMAKE_CURRENT_SOURCE_DIR}/basisset.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density_fitting.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ints.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/merge.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_qm
    PUBLIC 
    occ_core
    occ_io
    occ_gto
    Libint2::cxx
)
target_include_directories(occ_qm PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_qm PUBLIC cxx_std_17)

