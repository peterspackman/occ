add_library(occ_sht
    "${CMAKE_CURRENT_SOURCE_DIR}/sht.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_sht
    PUBLIC 
    occ_core
)

target_include_directories(occ_sht PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_sht PUBLIC cxx_std_17)
