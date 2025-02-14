add_subdirectory(3rdparty)
add_subdirectory(core)
add_subdirectory(disp)
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
add_subdirectory(cg)
add_subdirectory(driver)

add_subdirectory(isosurface)
add_subdirectory(main)

if("${WITH_PYTHON_BINDINGS}")

set(PYTHON_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/cg_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/crystal_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/dft_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/isosurface_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/qm_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occpy.cpp"
)

nanobind_add_module(_occpy
    NB_STATIC
    LTO
    STABLE_ABI
    ${PYTHON_BINDING_SOURCES}
)

target_include_directories(_occpy PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/python"
)

target_compile_definitions(_occpy PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(_occpy PUBLIC cxx_std_20)
target_link_libraries(_occpy PUBLIC 
    occ_main
)

install(TARGETS _occpy LIBRARY DESTINATION occpy)
endif()

# Update CMakeLists.txt to add WASM target
if(WASM_BUNDLE)
    set(WASM_BINDING_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/wasm/core_bindings.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/occ_wasm.cpp"
    )

    add_executable(occ_wasm ${WASM_BINDING_SOURCES})
    target_include_directories(occ_wasm PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/wasm"
    )
    target_compile_definitions(occ_wasm PRIVATE VERSION_INFO=${PROJECT_VERSION})
    target_compile_features(occ_wasm PUBLIC cxx_std_20)
    target_link_libraries(occ_wasm PUBLIC 
        occ_main
        embind
    )

    # Emscripten specific flags
    set_target_properties(occ_wasm PROPERTIES
        LINK_FLAGS "-s WASM=1 \
                   -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] \
                   -s EXPORTED_FUNCTIONS=['_malloc','_free'] \
                   -s ALLOW_MEMORY_GROWTH=1 \
                   -s NO_EXIT_RUNTIME=1 \
                   -s MODULARIZE=1 \
                   -s EXPORT_NAME='createOccModule'"
    )

    install(TARGETS occ_wasm 
        RUNTIME DESTINATION web
        LIBRARY DESTINATION web
    )
endif()
