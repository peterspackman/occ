add_library(occ_scrf
    "${CMAKE_CURRENT_SOURCE_DIR}/cosmo_kernel.cpp"
    ${OCC_SCRF_INCLUDE_FILES}
)

target_link_libraries(occ_scrf
    PUBLIC
    occ_core
)
target_include_directories(occ_scrf PUBLIC
    ${OCC_INCLUDE_DIR}
)

target_compile_features(occ_scrf PUBLIC cxx_std_20)
