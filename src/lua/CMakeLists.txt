add_library(occ_lua_bindings STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/occ_module.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/eigen_matrix.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/core_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/qm_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/descriptors_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/opt_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/interaction_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dma_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/isosurface_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/mults_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cg_bindings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/xtb_bindings.cpp"
)

target_include_directories(occ_lua_bindings PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_features(occ_lua_bindings PUBLIC cxx_std_20)

target_link_libraries(occ_lua_bindings
    PUBLIC
    luabridge3::luabridge3
    occ_core
    occ_xtb
    occ_crystal
    occ_qm
    occ_gto
    occ_io
    occ_opt
    occ_dft
    occ_xdm
    occ_descriptors
    occ_interaction
    occ_cg
    occ_dma
    occ_isosurface
    occ_mults
    occ_main
    occ_driver
    fmt::fmt
)

set_target_properties(occ_lua_bindings PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
