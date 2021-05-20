cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(occ LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
endif()

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost COMPONENTS graph)
find_package(Libint2 REQUIRED)
find_package(Libxc REQUIRED)
find_package(ZLIB REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fmt" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/scnlib" EXCLUDE_FROM_ALL)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use fmt from 3rdparty directory" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog" EXCLUDE_FROM_ALL)
set(GEMMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gemmi/include")
set(FASTOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Fastor/")
set(TOML_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/toml11/")

add_subdirectory(include)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
