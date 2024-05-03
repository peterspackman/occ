add_subdirectory(occ/core)
add_subdirectory(occ/crystal)
add_subdirectory(occ/descriptors)
add_subdirectory(occ/dft)
add_subdirectory(occ/geometry)
add_subdirectory(occ/gto)
add_subdirectory(occ/interaction)
add_subdirectory(occ/isosurface)
add_subdirectory(occ/io)
add_subdirectory(occ/main)
add_subdirectory(occ/qm)
add_subdirectory(occ/sht)
add_subdirectory(occ/slater)
add_subdirectory(occ/xdm)
add_subdirectory(occ/3rdparty)
if("${WITH_XTB}")
add_subdirectory(occ/xtb)
else()
set(OCC_XTB_INCLUDE_FILES "")
endif()

set(OCC_INCLUDE_FILES
    ${OCC_CORE_INCLUDE_FILES}
    ${OCC_CRYSTAL_INCLUDE_FILES}
    ${OCC_DESCRIPTORS_INCLUDE_FILES}
    ${OCC_DFT_INCLUDE_FILES}
    ${OCC_GEOMETRY_INCLUDE_FILES}
    ${OCC_GTO_INCLUDE_FILES}
    ${OCC_INTERACTION_INCLUDE_FILES}
    ${OCC_MAIN_INCLUDE_FILES}
    ${OCC_QM_INCLUDE_FILES}
    ${OCC_SHT_INCLUDE_FILES}
    ${OCC_SLATER_INCLUDE_FILES}
    ${OCC_XDM_INCLUDE_FILES}
    ${OCC_3RDPARTY_INCLUDE_FILES}
    ${OCC_XTB_INCLUDE_FILES}
    PARENT_SCOPE
)

set(OCC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
