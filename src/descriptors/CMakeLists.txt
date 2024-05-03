add_library(occ_descriptors
    "${CMAKE_CURRENT_SOURCE_DIR}/promolecule_shape.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/steinhardt.cpp"
    ${OCC_DESCRIPTORS_INCLUDE_FILES}
)

target_link_libraries(occ_descriptors
    PUBLIC 
    occ_sht
    occ_crystal
    occ_slater
)

target_include_directories(occ_descriptors PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_descriptors PUBLIC cxx_std_20)
