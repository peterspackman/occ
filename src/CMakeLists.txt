add_library(craso
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.cpp"
    ${CRASO_INCLUDE_FILES}
)

target_link_libraries(craso 
    PUBLIC 
    Eigen3::Eigen Boost::graph fmt::fmt 
    OpenMP::OpenMP_CXX ${BLAS_LIBRARIES}
    Libint2::cxx
)
target_include_directories(craso PUBLIC 
    ${CRASO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)
target_compile_features(craso PUBLIC cxx_std_17)


add_executable(hf "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")
target_link_libraries(hf PUBLIC craso)
