add_library(occ_disp
    "${CMAKE_CURRENT_SOURCE_DIR}/dftd4.cpp"
    ${OCC_DISP_INCLUDE_FILES}
)

target_link_libraries(occ_disp
    PUBLIC 
    occ_core
)

target_include_directories(occ_disp PUBLIC 
    ${OCC_INCLUDE_DIR} 
)

target_compile_features(occ_disp PUBLIC cxx_std_20)
