add_library(occ_slater
    "${CMAKE_CURRENT_SOURCE_DIR}/hirshfeld.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/promolecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/slaterbasis.cpp"
    ${OCC_SLATER_INCLUDE_FILES}
)

target_link_libraries(occ_slater
    PUBLIC 
    occ_core
    nlohmann_json::nlohmann_json
)

target_include_directories(occ_slater PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_slater PUBLIC cxx_std_20)
