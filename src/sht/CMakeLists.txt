add_library(
    occ_sht_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/clebsch.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/legendre.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/quadrature.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sht.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spherical_harmonics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wigner3j.cpp"
    ${OCC_SHT_INCLUDE_FILES}
)
add_library(occ_sht $<TARGET_OBJECTS:occ_sht_obj>)
target_include_directories(occ_sht_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_sht_obj PUBLIC occ_core)
target_link_libraries(occ_sht PUBLIC occ_core)

target_include_directories(occ_sht PUBLIC ${OCC_INCLUDE_DIR})

target_compile_features(occ_sht PUBLIC cxx_std_20)
