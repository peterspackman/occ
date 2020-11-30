add_library(tonto_slater
    "${CMAKE_CURRENT_SOURCE_DIR}/slaterbasis.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/thakkar.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_slater
    PUBLIC 
    tonto_core
)

target_include_directories(tonto_slater PUBLIC 
    ${TONTO_INCLUDE_DIR} 
)

target_compile_features(tonto_slater PUBLIC cxx_std_17)
