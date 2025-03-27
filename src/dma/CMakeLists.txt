add_library(occ_dma
    "${CMAKE_CURRENT_SOURCE_DIR}/dma.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gauss_hermite.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/quadrature.cpp"
)

target_link_libraries(occ_dma
    PUBLIC
    occ_core
    occ_gto
    occ_qm
    occ_dft
)
