add_library(tonto_solvent
    "${CMAKE_CURRENT_SOURCE_DIR}/cosmo.cpp"
    ${TONTO_SOLVENT_INCLUDE_FILES}
)

target_link_libraries(tonto_solvent
    PUBLIC 
    tonto_core
)
target_include_directories(tonto_solvent PUBLIC 
    ${TONTO_INCLUDE_DIR} 
)

target_compile_features(tonto_solvent PUBLIC cxx_std_17)
