add_library(occ_elastic_fit
    "${CMAKE_CURRENT_SOURCE_DIR}/monkhorst_pack.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/potentials.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/elastic_fitting.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/elastic_fit_json.cpp"
    ${OCC_ELASTIC_FIT_INCLUDE_FILES}
)

target_link_libraries(occ_elastic_fit
    PUBLIC 
    occ_core
    occ_crystal
    occ_io
)

target_include_directories(occ_elastic_fit PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_elastic_fit PUBLIC cxx_std_20)

