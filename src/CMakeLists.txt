add_subdirectory(3rdparty)
add_subdirectory(ints)
add_subdirectory(core)
add_subdirectory(disp)
add_subdirectory(crystal)
add_subdirectory(dma)
add_subdirectory(interaction)
add_subdirectory(geometry)
add_subdirectory(gto)
add_subdirectory(io)
add_subdirectory(elastic_fit)
add_subdirectory(qm)
add_subdirectory(numint)
add_subdirectory(dft)
add_subdirectory(sht)
add_subdirectory(slater)
add_subdirectory(descriptors)
add_subdirectory(solvent)
add_subdirectory(xdm)
add_subdirectory(xtb)
add_subdirectory(cg)
add_subdirectory(opt)
add_subdirectory(driver)

add_subdirectory(isosurface)
add_subdirectory(main)

if("${WITH_PYTHON_BINDINGS}")

# Include Python bindings utilities
include(PythonBindings)

set(PYTHON_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/python/cg_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/crystal_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/descriptors_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/dft_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/dma_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/isosurface_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/interaction_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/python/opt_bindings.cpp"
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
    occ_opt
    occ_descriptors
)

install(TARGETS _occpy LIBRARY DESTINATION occpy)

# Generate Python stub files for type hints and documentation
add_nanobind_stubs(_occpy _occpy ${CMAKE_CURRENT_BINARY_DIR}/stubs)

endif()

# JavaScript/WASM bindings for Emscripten
if(ENABLE_JS_BINDINGS)

set(JS_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/js/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/qm_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/dft_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/opt_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/isosurface_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/cube_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/crystal_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/descriptors_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/dma_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/js/volume_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/occjs.cpp"
)

add_executable(occjs ${JS_BINDING_SOURCES})

target_include_directories(occjs PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/js"
)

target_compile_definitions(occjs PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_compile_features(occjs PUBLIC cxx_std_20)
target_compile_options(occjs PRIVATE -fexceptions -pthread)

# Link against the core OCC libraries
target_link_libraries(occjs PUBLIC
    occ_core
    occ_qm
    occ_gto
    occ_io
    occ_dft
    occ_dma
    occ_driver
    occ_isosurface
    occ_opt
    occ_descriptors
)

# Emscripten-specific linker flags for JavaScript bindings
# Note: static_partitioner in XDM code avoids Firefox WASM threading issues
set_target_properties(occjs PROPERTIES
    LINK_FLAGS "-s MODULARIZE=1 -s EXPORT_NAME='createOccModule' -s EXPORT_ES6=1 -s ENVIRONMENT='web,worker,node' -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=1GB -s STACK_SIZE=8MB -s DISABLE_EXCEPTION_CATCHING=0 -fexceptions -pthread --bind --preload-file ${CMAKE_SOURCE_DIR}/share@/"
)

# Set output file extension
set_target_properties(occjs PROPERTIES SUFFIX ".js")

# WASM executable version of occ CLI (standalone, non-modularized)
add_executable(occ.wasm "${CMAKE_CURRENT_SOURCE_DIR}/main/occ.cpp")

target_compile_features(occ.wasm PUBLIC cxx_std_20)
target_compile_options(occ.wasm PRIVATE -O3 -DNDEBUG -fexceptions -pthread)

target_link_libraries(occ.wasm PUBLIC
    occ_main
)

# Emscripten-specific linker flags for CLI executable (matching occjs for performance)
set_target_properties(occ.wasm PROPERTIES
    LINK_FLAGS "-O3 -DNDEBUG -flto -s ENVIRONMENT='web,worker,node' -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=1GB -s STACK_SIZE=8MB -s DISABLE_EXCEPTION_CATCHING=0 -fexceptions -pthread -s EXIT_RUNTIME=0 -s INVOKE_RUN=0 -s FORCE_FILESYSTEM=1 -s EXPORTED_RUNTIME_METHODS=['callMain','FS'] --preload-file ${CMAKE_SOURCE_DIR}/share@/ -s ASSERTIONS=0"
)

# Set output file extension (will generate occ.js and occ.wasm)
set_target_properties(occ.wasm PROPERTIES OUTPUT_NAME "occ" SUFFIX ".js")

endif()
