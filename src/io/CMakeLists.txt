add_library(occ_io
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkwriter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/inputfile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/moldenreader.cpp"
    ${OCC_INCLUDE_FILES}
)

target_link_libraries(occ_io
    PUBLIC 
    occ_core
    occ_qm
    toml11::toml11
)
target_include_directories(occ_io PUBLIC 
    ${OCC_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(occ_io PUBLIC cxx_std_17)
