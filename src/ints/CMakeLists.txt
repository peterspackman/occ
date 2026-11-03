add_library(occ_ints_obj OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/esp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ecp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ecp_math.cpp"
)
add_library(occ_ints $<TARGET_OBJECTS:occ_ints_obj>)
target_include_directories(occ_ints_obj PUBLIC ${OCC_INCLUDE_DIR})
target_link_libraries(occ_ints_obj PUBLIC Eigen3::Eigen occ_gto)

target_include_directories(occ_ints PUBLIC ${OCC_INCLUDE_DIR})

target_link_libraries(occ_ints PUBLIC Eigen3::Eigen occ_gto)

target_compile_features(occ_ints PUBLIC cxx_std_20)

# The ESP kernels pass vector types between always-inlined helpers, so GCC's
# note that their ABI differs between targets does not apply.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        "${CMAKE_CURRENT_SOURCE_DIR}/esp.cpp"
        PROPERTIES COMPILE_OPTIONS "-Wno-psabi"
    )
endif()
