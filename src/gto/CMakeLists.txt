add_library(occ_gto_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/json.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/rotation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shell.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sto3g_basis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/io/json_basis.cpp"
    ${OCC_GTO_INCLUDE_FILES}
)
add_library(occ_gto $<TARGET_OBJECTS:occ_gto_obj>)
target_include_directories(occ_gto_obj PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR})
target_link_libraries(occ_gto_obj PUBLIC occ_core gau2grid_static nlohmann_json::nlohmann_json)

target_link_libraries(
    occ_gto
    PUBLIC occ_core gau2grid_static nlohmann_json::nlohmann_json
)
target_include_directories(
    occ_gto
    PUBLIC ${OCC_INCLUDE_DIR} ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_gto PUBLIC cxx_std_20)
