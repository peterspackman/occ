function(create_solvent_test arg)
get_filename_component(test_name ${arg} NAME_WE)
add_executable(${test_name} ${arg} ${TEST_MAIN})
# occ_io is linked so xyz strings etc. can be read
target_link_libraries(${test_name} occ_solvent occ_io)
target_include_directories(${test_name} PRIVATE ${TEST_INCLUDE_DIR})
add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

create_solvent_test(solvent_tests.cpp)
