add_library(occ_geometry
    "${CMAKE_CURRENT_SOURCE_DIR}/icosphere_mesh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/index_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/marching_cubes_data.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/morton.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/quickhull.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/wulff.cpp"
    ${OCC_GEOMETRY_INCLUDE_FILES}
)

target_link_libraries(occ_geometry
    PUBLIC 
    occ_core
)
target_include_directories(occ_geometry PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_geometry PUBLIC cxx_std_20)

