add_library(tonto_gto
    "${CMAKE_CURRENT_SOURCE_DIR}/density.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/gto.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_gto
    PUBLIC 
    tonto_core
    tonto_qm
)
target_include_directories(tonto_gto PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(tonto_gto PUBLIC cxx_std_17)

