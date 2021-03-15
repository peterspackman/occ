add_subdirectory(tonto/core)
add_subdirectory(tonto/crystal)
add_subdirectory(tonto/dft)
add_subdirectory(tonto/geometry)
add_subdirectory(tonto/gto)
add_subdirectory(tonto/interaction)
add_subdirectory(tonto/io)
add_subdirectory(tonto/qm)
add_subdirectory(tonto/slater)
add_subdirectory(tonto/3rdparty)

set(TONTO_INCLUDE_FILES
    ${TONTO_CORE_INCLUDE_FILES}
    ${TONTO_CRYSTAL_INCLUDE_FILES}
    ${TONTO_DFT_INCLUDE_FILES}
    ${TONTO_GTO_INCLUDE_FILES}
    ${TONTO_INTERACTION_INCLUDE_FILES}
    ${TONTO_QM_INCLUDE_FILES}
    ${TONTO_SLATER_INCLUDE_FILES}
    ${TONTO_3RDPARTY_INCLUDE_FILES}
    PARENT_SCOPE
)

set(TONTO_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
