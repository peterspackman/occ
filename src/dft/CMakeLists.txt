add_library(occ_dft_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/dft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_kernels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_gradient_kernels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_gradients.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_method.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/functional.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hirshfeld.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/voronoi_charges.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nonlocal_correlation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xc_potential_matrix.cpp"
    ${OCC_DFT_INCLUDE_FILES}
)
add_library(occ_dft $<TARGET_OBJECTS:occ_dft_obj>)
target_include_directories(occ_dft_obj PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR})
target_link_libraries(occ_dft_obj PUBLIC occ_core occ_qm occ_io occ_numint occ_slater Libxc::xc)

target_link_libraries(
    occ_dft
    PUBLIC occ_core occ_qm occ_io occ_numint occ_slater Libxc::xc
)
target_include_directories(
    occ_dft
    PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_dft PUBLIC cxx_std_20)
