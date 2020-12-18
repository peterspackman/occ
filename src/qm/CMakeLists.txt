add_library(tonto_qm
    "${CMAKE_CURRENT_SOURCE_DIR}/basisset.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density_fitting.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ints.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/merge.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_qm
    PUBLIC 
    tonto_core
    tonto_io
    Libint2::cxx
)
target_include_directories(tonto_qm PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(tonto_qm PUBLIC cxx_std_17)

