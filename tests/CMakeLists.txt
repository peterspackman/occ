macro(create_catch_test arg)
get_filename_component(test_name ${arg} NAME_WE)
add_executable(${test_name} ${arg})
target_include_directories(${test_name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CRASO_INCLUDE_DIR}"
)
target_link_libraries(${test_name} craso Eigen3::Eigen fmt::fmt Boost::graph)
target_compile_features(${test_name} PRIVATE cxx_std_17)
add_test(NAME ${test_name} COMMAND ${test_name})
endmacro()

create_catch_test(test_element.cpp)
create_catch_test(test_symmetry_operation.cpp)
