add_library(tonto_geometry
    "${CMAKE_CURRENT_SOURCE_DIR}/index_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/morton.cpp"
    ${TONTO_CORE_INCLUDE_FILES}
)

target_link_libraries(tonto_geometry
    PUBLIC 
    Boost::graph
    Eigen3::Eigen 
    fmt::fmt
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    scn::scn
    spdlog::spdlog
)
target_include_directories(tonto_geometry PUBLIC 
    ${TONTO_INCLUDE_DIR} 
)

target_compile_features(tonto_geometry PUBLIC cxx_std_17)

