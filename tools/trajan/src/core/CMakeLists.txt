add_library(
  trajan_core
  "${CMAKE_CURRENT_SOURCE_DIR}/rdf.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/frame.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/molecule.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/neigh.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/log.cpp" 
  "${CMAKE_CURRENT_SOURCE_DIR}/atom.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/trajectory.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/topology.cpp"
)

target_link_libraries(trajan_core PUBLIC fmt::fmt spdlog::spdlog Eigen3::Eigen unordered_dense::unordered_dense occ_core occ_crystal)

target_include_directories(trajan_core PUBLIC "${TRAJAN_INCLUDE_DIR}")

target_compile_features(trajan_core PUBLIC cxx_std_20)
