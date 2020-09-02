add_executable(craso
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/element.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/periodicbondgraph.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.cpp"
    ${CRASO_INCLUDE_FILES}
)

target_link_libraries(craso PRIVATE Eigen3::Eigen Boost::graph)
target_include_directories(craso PRIVATE 
    ${CRASO_INCLUDE_DIR} ${FMT_INCLUDE_DIR}
    ${GEMMI_INCLUDE_DIR}
)
target_compile_features(craso PRIVATE cxx_std_17)
