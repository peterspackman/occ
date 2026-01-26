add_library(occ_ints
    "${CMAKE_CURRENT_SOURCE_DIR}/esp.cpp"
)

target_include_directories(occ_ints PUBLIC
    ${OCC_INCLUDE_DIR}
)

target_link_libraries(occ_ints
    PUBLIC
    Eigen3::Eigen
    occ_gto
)

target_compile_features(occ_ints PUBLIC cxx_std_20)
