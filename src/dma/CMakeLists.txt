add_library(occ_dma_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/add_qlm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/binomial.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dma.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gauss_hermite.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_multipole_shifter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_multipole_calculator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mult.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole_calculator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/multipole_shifter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/solid_harmonics.cpp"
)
add_library(occ_dma $<TARGET_OBJECTS:occ_dma_obj>)
target_include_directories(occ_dma_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_dma_obj PUBLIC occ_core occ_gto occ_qm occ_dft)

target_include_directories(occ_dma PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_dma PUBLIC occ_core occ_gto occ_qm occ_dft)
