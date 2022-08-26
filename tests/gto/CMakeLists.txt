function(create_gto_test arg)
get_filename_component(test_name ${arg} NAME_WE)
add_executable(${test_name} ${arg})
target_link_libraries(${test_name} occ_gto occ_dft Catch2::Catch2WithMain)
add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

create_gto_test(gto_tests.cpp)
