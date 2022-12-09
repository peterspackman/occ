add_library(occ_xtb
    STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/xtb_wrapper.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_xtb
    PUBLIC 
    occ_core
    occ_qm
    occ_dft
    ${tblite_STATIC_LIBRARIES}
    gomp
)

target_include_directories(occ_xtb PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
    ${tblite_STATIC_INCLUDE_DIRS}
)
message(STATUS "TBLITE include: ${tblite_STATIC_INCLUDE_DIRS}")
message(STATUS "TBLITE libs: ${tblite_STATIC_LIBRARIES}")

target_compile_features(occ_xtb PUBLIC cxx_std_17)

