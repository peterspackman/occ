add_library(occ_descriptors_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/promolecule_shape.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/steinhardt.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pdd_amd.cpp"
    ${OCC_DESCRIPTORS_INCLUDE_FILES}
)
add_library(occ_descriptors $<TARGET_OBJECTS:occ_descriptors_obj>)
target_include_directories(occ_descriptors_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_descriptors_obj PUBLIC occ_sht occ_crystal occ_slater)

target_link_libraries(occ_descriptors PUBLIC occ_sht occ_crystal occ_slater)

target_include_directories(occ_descriptors PUBLIC ${OCC_INCLUDE_DIR})

target_compile_features(occ_descriptors PUBLIC cxx_std_20)
