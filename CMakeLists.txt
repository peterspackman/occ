cmake_minimum_required(VERSION 3.16.0)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(occ LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(BUILD_FPIC "Build with position independent code" OFF)
    option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
    option(WITH_CLANG_TIDY "Enable clang tidy" OFF)
    option(USE_SYSTEM_LIBINT2 "Use Libint2 installed on system" ON)
    option(USE_SYSTEM_LIBXC "Use LibXC installed on system" ON)
endif()

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(Boost CONFIG COMPONENTS graph)
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)

if("${USE_SYSTEM_LIBINT2}")
    find_package(Libint2 CONFIG REQUIRED)
endif()

if("${USE_SYSTEM_LIBXC}")
    find_package(Libxc CONFIG REQUIRED)
endif()


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
add_subdirectory(3rdparty)

set(GEMMI_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/gemmi/include")
set(FASTOR_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Fastor/")

add_subdirectory(include)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
if("${WITH_CLANG_TIDY}")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy;clang-diagnostic-*,clang-analyzer-*,-*,bugprone*,modernize*,performance*,-modernize-pass-by-value,-modernize-use-auto,-modernize-use-using")
endif()
