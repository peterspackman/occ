cmake_minimum_required(VERSION 3.10.0)

project(craso LANGUAGES CXX)

find_package(Eigen3 CONFIG REQUIRED)
find_package(Boost COMPONENTS graph)

add_subdirectory(3rdparty/fmt)
set(GEMMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gemmi/include")

add_subdirectory(include)
add_subdirectory(src)
