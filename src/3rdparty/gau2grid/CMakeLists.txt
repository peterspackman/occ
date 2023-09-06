add_library(gau2grid_static
    STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_deriv1.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_deriv2.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_deriv3.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_helper.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_orbital.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_phi.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/gau2grid_transform.c"
)

set(OCC_3RDPARTY_GAU2GRID_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(gau2grid_static
    PUBLIC 
    ${OCC_3RDPARTY_GAU2GRID_INCLUDE_DIR}
)

if("${GG_NO_PRAGMA}")
  target_compile_definitions(gau2grid_static PRIVATE $<BUILD_INTERFACE:__GG_NO_PRAGMA> )
endif()
