add_library(occ_ints_obj OBJECT "${CMAKE_CURRENT_SOURCE_DIR}/esp.cpp")
add_library(occ_ints $<TARGET_OBJECTS:occ_ints_obj>)
target_include_directories(occ_ints_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_ints_obj PUBLIC Eigen3::Eigen occ_gto)

target_include_directories(occ_ints PUBLIC ${OCC_INCLUDE_DIR})

target_link_libraries(occ_ints PUBLIC Eigen3::Eigen occ_gto)

target_compile_features(occ_ints PUBLIC cxx_std_20)
