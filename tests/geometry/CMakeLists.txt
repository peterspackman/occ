function(create_geometry_test arg)
get_filename_component(test_name ${arg} NAME_WE)
add_executable(${test_name} ${arg} ${TEST_MAIN})
target_link_libraries(${test_name} occ_geometry)
target_include_directories(${test_name} PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

create_geometry_test(geometry_tests.cpp)
