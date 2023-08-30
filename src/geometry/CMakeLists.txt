add_library(occ_geometry
    "${CMAKE_CURRENT_SOURCE_DIR}/index_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/morton.cpp"
    ${OCC_CORE_INCLUDE_FILES}
)

target_link_libraries(occ_geometry
    PUBLIC 
    Eigen3::Eigen 
    fmt::fmt
    unordered_dense::unordered_dense
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    scn::scn
    spdlog::spdlog
)
target_include_directories(occ_geometry PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_geometry PUBLIC cxx_std_17)

