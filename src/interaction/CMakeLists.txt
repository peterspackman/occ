add_library(tonto_interaction
    "${CMAKE_CURRENT_SOURCE_DIR}/disp.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pairinteraction.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/polarization.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_interaction
    PUBLIC 
    tonto_core
    tonto_crystal
    tonto_qm
    tonto_io
)
target_include_directories(tonto_interaction PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(tonto_interaction PUBLIC cxx_std_17)
