add_subdirectory(3rdparty)
add_subdirectory(core)
add_subdirectory(disp)
add_subdirectory(crystal)
add_subdirectory(dma)
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
    "${CMAKE_CURRENT_SOURCE_DIR}/python/dma_bindings.cpp"
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

# JavaScript/WASM bindings for Emscripten
if(ENABLE_JS_BINDINGS)

set(JS_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/js/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/qm_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occjs.cpp"
)

add_executable(occjs ${JS_BINDING_SOURCES})

target_include_directories(occjs PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/js"
)

target_compile_definitions(occjs PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(occjs PUBLIC cxx_std_20)

# Link against the core OCC libraries
target_link_libraries(occjs PUBLIC 
    occ_core
    occ_qm
    occ_gto
    occ_io
    occ_dft
)

# Emscripten-specific linker flags for JavaScript bindings
set_target_properties(occjs PROPERTIES
    LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME='createOccModule' -s EXPORT_ES6=1 -s ENVIRONMENT='web,worker,node' -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=1GB -s STACK_SIZE=5MB -s DISABLE_EXCEPTION_CATCHING=0 --bind"
)

# Set output file extension
set_target_properties(occjs PROPERTIES SUFFIX ".js")

endif()
