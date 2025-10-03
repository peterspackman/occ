add_library(trajan_io "${CMAKE_CURRENT_SOURCE_DIR}/pdb.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/dcd.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/file_handler.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/selection.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/xyz.cpp"
                     "${CMAKE_CURRENT_SOURCE_DIR}/xtc.cpp"
)

target_link_libraries(trajan_io PUBLIC Eigen3::Eigen fmt::fmt spdlog::spdlog unordered_dense::unordered_dense occ_crystal xtc::xtc)

target_include_directories(trajan_io PUBLIC "${TRAJAN_INCLUDE_DIR}")

target_compile_features(trajan_io PUBLIC cxx_std_20)
