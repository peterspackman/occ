add_library(trajan_main
  "${CMAKE_CURRENT_SOURCE_DIR}/trajan_rdf.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/trajan_topology.cpp"
  #  "${CMAKE_CURRENT_SOURCE_DIR}/trajan_opt.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/version.cpp"
  ${TRAJAN_MAIN_INCLUDE_FILES}
)

target_link_libraries(trajan_main PUBLIC trajan_core trajan_io CLI11::CLI11)
# target_link_libraries(trajan_main PUBLIC trajan_core trajan_io trajan_energy CLI11::CLI11)

target_include_directories(trajan_main PUBLIC 
    ${TRAJAN_INCLUDE_DIR} 
)

add_executable(
  trajan
  "${CMAKE_CURRENT_SOURCE_DIR}/trajan.cpp")

target_link_libraries(trajan
    PUBLIC
    trajan_main
)
set_target_properties(trajan
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_compile_features(trajan PUBLIC cxx_std_20)
