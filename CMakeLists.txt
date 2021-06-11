cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(occ LANGUAGES CXX)

option(BUILD_FPIC "Build with position independent code" OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
endif()

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost CONFIG COMPONENTS graph)
find_package(Libint2 CONFIG REQUIRED)
find_package(Libxc CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PCMSolver CONFIG COMPONENTS static)

set(OCC_DEFINITIONS "")
if("${PCMSolver_FOUND}")
    message(STATUS "PCMSolver found, building library with PCMSolver support")
    message(STATUS "PCMSolver_LIBRARIES: ${PCMSolver_LIBRARIES}")
    message(STATUS "PCMSolver_INCLUDE_DIR: ${PCMSolver_INCLUDE_DIR}")
    set(OCC_DEFINITIONS "${OCC_DEFINITIONS} ${PCMSolver_DEFINITIONS}")
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fmt" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/scnlib" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gau2grid" EXCLUDE_FROM_ALL)

set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use fmt from 3rdparty directory" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/spdlog" EXCLUDE_FROM_ALL)
set(GEMMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gemmi/include")
set(FASTOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Fastor/")
set(TOML_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/toml11/")

message(STATUS "OCC_DEFINITIONS: ${PCMSolver_DEFINITIONS}")
add_subdirectory(include)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
