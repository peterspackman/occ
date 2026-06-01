add_library(occ_cc
    "${CMAKE_CURRENT_SOURCE_DIR}/thc.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/integrals.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ccsd.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/triples.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/uccsd_so.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/uintegrals.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/uccsd.cpp"
    ${OCC_CC_INCLUDE_FILES}
)

target_link_libraries(occ_cc
    PUBLIC
    occ_qm
    occ_correlation
    occ_numint
    occ_ints
)
target_include_directories(occ_cc PUBLIC
    ${OCC_INCLUDE_DIR}
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_cc PUBLIC cxx_std_20)
