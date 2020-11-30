add_library(tonto_crystal
    "${CMAKE_CURRENT_SOURCE_DIR}/crystal.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/spacegroup.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/symmetryoperation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/unitcell.cpp"
    ${TONTO_INCLUDE_FILES}
)

target_link_libraries(tonto_crystal
    PUBLIC 
    tonto_core
)
target_include_directories(tonto_crystal PUBLIC 
    ${TONTO_INCLUDE_DIR} 
    ${GEMMI_INCLUDE_DIR}
)

target_compile_features(tonto_crystal PUBLIC cxx_std_17)
