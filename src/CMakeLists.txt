add_library(tonto STATIC
    "${CMAKE_CURRENT_SOURCE_DIR}/basisset.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density_fitting.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/density_functional.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dft_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/disp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/hf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/inputfile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ints.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lebedev.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/linear_algebra.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/moldenreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/optimize.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pairinteraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/parallel.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/polarization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/scf.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/slaterbasis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/table.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/thakkar.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/timings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wavefunction.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto 
    PUBLIC 
    Eigen3::Eigen Boost::graph
    ZLIB::ZLIB
    fmt::fmt spdlog::spdlog
    OpenMP::OpenMP_CXX ${BLAS_LIBRARIES}
    Libint2::cxx ${LAPACK_LIBRARIES}
    Libxc::xc
    scn::scn
)
target_include_directories(tonto PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
    ${FASTOR_INCLUDE_DIR}
    ${TOML_INCLUDE_DIR}
)
target_compile_features(tonto PUBLIC cxx_std_17)


add_executable(tontox "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(tontox PUBLIC tonto)

add_executable(ce "${CMAKE_CURRENT_SOURCE_DIR}/ce_main.cpp")
target_link_libraries(ce PUBLIC tonto)

add_executable(make_radial_interpolators "${CMAKE_CURRENT_SOURCE_DIR}/make_radial_interpolators.cpp")
target_link_libraries(make_radial_interpolators PUBLIC tonto)
