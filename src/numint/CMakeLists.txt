add_library(occ_numint_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_settings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_types.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid_utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lebedev.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/molecular_grid_points.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spatial_grid_hierarchy.cpp"
    ${OCC_NUMINT_INCLUDE_FILES}
)
add_library(occ_numint $<TARGET_OBJECTS:occ_numint_obj>)
target_include_directories(occ_numint_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_numint_obj PUBLIC occ_core occ_gto)

target_link_libraries(occ_numint PUBLIC occ_core occ_gto)
target_include_directories(occ_numint PUBLIC ${OCC_INCLUDE_DIR})

target_compile_features(occ_numint PUBLIC cxx_std_20)
