cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

project(occ
    VERSION 0.3
    LANGUAGES CXX
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
        FATAL_ERROR
        "In-source builds are not supported. Please make a new directory (i.e. build directory) and run cmake from there."
    )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(BUILD_FPIC "Build with position independent code" OFF)
    option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
    option(WITH_CLANG_TIDY "Enable clang tidy" OFF)
    option(USE_SYSTEM_BOOST "Use Boost installed on system" ON)
    option(USE_SYSTEM_LIBINT2 "Use Libint2 installed on system" ON)
    option(USE_SYSTEM_LIBXC "Use LibXC installed on system" ON)
#    option(USE_SYSTEM_BLAS "Use BLAS/LAPACK installed on system" ON)
    option(USE_SYSTEM_ZLIB "Use ZLIB installed on system" ON)
    option(USE_SYSTEM_EIGEN "Use EIGEN installed on system" ON)
endif()

find_package(Threads REQUIRED)

if("${USE_SYSTEM_BOOST}")
    find_package(Boost CONFIG COMPONENTS graph)
endif()

if("${USE_SYSTEM_EIGEN}")
    find_package(Eigen3 CONFIG REQUIRED)
endif()

if("${USE_SYSTEM_ZLIB}")
    find_package(ZLIB REQUIRED)
endif()

#if("${USE_SYSTEM_BLAS}")
#    find_package(LAPACK REQUIRED)
#    find_package(BLAS REQUIRED)
#endif()

if("${USE_SYSTEM_LIBINT2}")
    find_package(Libint2 CONFIG REQUIRED)
endif()

if("${USE_SYSTEM_LIBXC}")
    find_package(Libxc CONFIG REQUIRED)
endif()


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
add_subdirectory(3rdparty)
add_subdirectory(include)
add_subdirectory(src)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if("${WITH_CLANG_TIDY}")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy;clang-diagnostic-*,clang-analyzer-*,-*,bugprone*,modernize*,performance*,-modernize-pass-by-value,-modernize-use-auto,-modernize-use-using")
endif()
