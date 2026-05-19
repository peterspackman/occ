add_library(occ_disp
    "${CMAKE_CURRENT_SOURCE_DIR}/d3.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/d3_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/d4.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/d4_data.cpp"
    ${OCC_DISP_INCLUDE_FILES}
)

target_link_libraries(occ_disp
    PUBLIC
    occ_core
    nlohmann_json::nlohmann_json
)

target_include_directories(occ_disp PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_disp PUBLIC cxx_std_20)
