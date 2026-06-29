add_library(occ_geometry_obj
    OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/icosphere_mesh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/morton.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/quickhull.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/volume_grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wulff.cpp"
    ${OCC_GEOMETRY_INCLUDE_FILES}
)
add_library(occ_geometry $<TARGET_OBJECTS:occ_geometry_obj>)
target_include_directories(occ_geometry_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_geometry_obj PUBLIC occ_core)

target_link_libraries(occ_geometry PUBLIC occ_core)
target_include_directories(occ_geometry PUBLIC ${OCC_INCLUDE_DIR})

target_compile_features(occ_geometry PUBLIC cxx_std_20)
