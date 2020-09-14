cmake_minimum_required(VERSION 3.10.0)

project(craso LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
endif()

find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost COMPONENTS graph)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/fmt")
set(BUILD_SHARED_LIBS OFF)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/libcint")
set(GEMMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gemmi/include")
set(LIBCINT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/3rdparty/libcint/include")

add_subdirectory(include)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    add_subdirectory(tests)
endif()
