add_library(tonto STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density_functional.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ints.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/parallel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto 
    PUBLIC 
    Eigen3::Eigen Boost::graph
    fmt::fmt spdlog::spdlog
    OpenMP::OpenMP_CXX ${BLAS_LIBRARIES}
    Libint2::cxx LAPACK::LAPACK
    Libxc::xc
    numgrid-objects
)
target_include_directories(tonto PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)
target_compile_features(tonto PUBLIC cxx_std_17)


add_executable(hf "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(hf PUBLIC tonto)
