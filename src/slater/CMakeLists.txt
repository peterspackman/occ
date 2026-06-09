add_library(occ_slater_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/hirshfeld.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/promolecule.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/slaterbasis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/thakkar_basis.cpp"
    ${OCC_SLATER_INCLUDE_FILES}
)
add_library(occ_slater $<TARGET_OBJECTS:occ_slater_obj>)
target_include_directories(occ_slater_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_slater_obj PUBLIC occ_core nlohmann_json::nlohmann_json)

target_link_libraries(occ_slater PUBLIC occ_core nlohmann_json::nlohmann_json)

target_include_directories(occ_slater PUBLIC ${OCC_INCLUDE_DIR})

target_compile_features(occ_slater PUBLIC cxx_std_20)
