cpmaddpackage(NAME catch GITHUB_REPOSITORY "catchorg/Catch2" GIT_TAG v3.5.2)

list(APPEND CMAKE_MODULE_PATH "${catch_SOURCE_DIR}/extras")
include(CTest)
include(Catch)

add_executable(test_trajan "${CMAKE_CURRENT_SOURCE_DIR}/trajan_tests.cpp")
target_link_libraries(test_trajan PRIVATE trajan_core trajan_io Catch2::Catch2WithMain)
catch_discover_tests(test_trajan)

