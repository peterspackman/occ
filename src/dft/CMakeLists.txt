add_library(occ_dft
    "${CMAKE_CURRENT_SOURCE_DIR}/dft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_kernels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_gradient_kernels.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_gradients.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_method.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/functional.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hirshfeld.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/voronoi_charges.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lebedev.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid_points.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nonlocal_correlation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/seminumerical_exchange.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xc_potential_matrix.cpp"
    ${OCC_DFT_INCLUDE_FILES}
)

target_link_libraries(occ_dft
    PUBLIC 
    occ_core
    occ_qm
    occ_io
    Libxc::xc
)
target_include_directories(occ_dft PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_dft PUBLIC cxx_std_20)

