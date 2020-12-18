add_library(tonto_dft
    "${CMAKE_CURRENT_SOURCE_DIR}/dft.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/functional.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/grid.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/lebedev.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_dft
    PUBLIC 
    tonto_core
    tonto_qm
    Libxc::xc
)
target_include_directories(tonto_dft PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(tonto_dft PUBLIC cxx_std_17)

