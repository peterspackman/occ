function(create_qm_test arg)
get_filename_component(test_name ${arg} NAME_WE)
add_executable(${test_name} ${arg})
target_link_libraries(${test_name} occ_qm Catch2::Catch2WithMain)
add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

create_qm_test(qm_tests.cpp)
