add_library(tonto_core
    "${CMAKE_CURRENT_SOURCE_DIR}/eem.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_algebra.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/optimize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/parallel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/table.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/timings.cpp"
    ${TONTO_CORE_INCLUDE_FILES}
)

target_link_libraries(tonto_core
    PUBLIC 
    Boost::graph
    Eigen3::Eigen 
    fmt::fmt
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
    scn::scn
    spdlog::spdlog
    ZLIB::ZLIB
)
target_include_directories(tonto_core PUBLIC 
    ${TONTO_INCLUDE_DIR} 
)

target_compile_features(tonto_core PUBLIC cxx_std_17)

