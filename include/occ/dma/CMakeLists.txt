set(HEADERS
    add_qlm.h
    binomial.h
    dma.h
    gauss_hermite.h
    linear_multipole_shifter.h
    linear_multipole_calculator.h
    mult.h
    multipole_calculator.h
    multipole_shifter.h
    solid_harmonics.h
)

set(OCC_DMA_INCLUDE_FILES ${HEADERS} PARENT_SCOPE)

add_library(dma_headers INTERFACE)
target_include_directories(dma_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../../)
add_library(occ::dma_headers ALIAS dma_headers)
install(FILES ${HEADERS} DESTINATION include/occ/dma)
