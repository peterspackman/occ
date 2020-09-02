set(CX_INCLUDE_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/kdtree.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/logger.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/periodicbondgraph.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/util.h"
    # third party
    "${CMAKE_CURRENT_SOURCE_DIR}/nanoflann.hpp"
    PARENT_SCOPE
)

set(CX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
