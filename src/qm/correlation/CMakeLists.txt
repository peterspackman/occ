add_library(occ_correlation_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/df_integrals.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mo_integral_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mp2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/post_hf_method.cpp"
    ${OCC_CORRELATION_INCLUDE_FILES}
)
add_library(occ_correlation $<TARGET_OBJECTS:occ_correlation_obj>)
target_include_directories(occ_correlation_obj PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR})
target_link_libraries(occ_correlation_obj PUBLIC occ_qm occ_ints)

target_link_libraries(occ_correlation PUBLIC occ_qm occ_ints)
target_include_directories(
    occ_correlation
    PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_correlation PUBLIC cxx_std_20)
