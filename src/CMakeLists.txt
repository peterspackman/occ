add_subdirectory(3rdparty)
add_subdirectory(core)
add_subdirectory(crystal)
add_subdirectory(interaction)
add_subdirectory(geometry)
add_subdirectory(gto)
add_subdirectory(io)
add_subdirectory(qm)
add_subdirectory(dft)
add_subdirectory(sht)
add_subdirectory(slater)
add_subdirectory(descriptors)
add_subdirectory(solvent)
add_subdirectory(xdm)
add_subdirectory(xtb)

add_subdirectory(isosurface)
add_subdirectory(main)

if("${WITH_PYTHON_BINDINGS}")
nanobind_add_module(_occpy
    NB_STATIC
    LTO

    # Target the stable ABI for python 3.12+
    STABLE_ABI

    "${CMAKE_CURRENT_SOURCE_DIR}/occpy.cpp")
target_compile_definitions(_occpy PRIVATE VERSION_INFO=${PROJECT_VERSION})

target_compile_features(_occpy PUBLIC cxx_std_20)

target_link_libraries(_occpy PUBLIC 
    occ_main
)

install(TARGETS _occpy LIBRARY DESTINATION occpy)
endif()
