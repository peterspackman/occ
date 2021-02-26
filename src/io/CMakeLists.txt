add_library(tonto_io
    "${CMAKE_CURRENT_SOURCE_DIR}/cifparser.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/conversion.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkreader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/fchkwriter.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/inputfile.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/moldenreader.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_io
    PUBLIC 
    tonto_core
    tonto_qm
)
target_include_directories(tonto_io PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
    ${TOML_INCLUDE_DIR}
)

target_compile_features(tonto_io PUBLIC cxx_std_17)
