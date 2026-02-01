add_library(
  trajan_energy
  "${CMAKE_CURRENT_SOURCE_DIR}/xtb.cpp"
)

if(TARGET xtb::xtb)
  message(STATUS "✓ xtb::xtb is available")
else()
  message(FATAL_ERROR "✗ xtb::xtb is NOT available!")
endif()

target_link_libraries(trajan_energy PUBLIC occ_core occ_crystal xtb::xtb)

target_include_directories(trajan_energy PUBLIC "${TRAJAN_INCLUDE_DIR}")

target_compile_features(trajan_energy PUBLIC cxx_std_20)
