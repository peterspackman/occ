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

if(SKBUILD)
  # Scikit-Build does not add your site-packages to the search path
  # automatically, so we need to add it _or_ the pybind11 specific directory
  # here.
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE _tmp_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
  list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")
  set(WITH_PYTHON_BINDINGS ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(BUILD_FPIC "Build with position independent code" OFF)
    option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
    option(WITH_CLANG_TIDY "Enable clang tidy" OFF)
    option(WITH_PYTHON_BINDINGS "Build python bindings" OFF)
    option(BUILD_DOCS "Build documentation using Doxygen" OFF)
    option(USE_SYSTEM_LIBXC "Use LibXC installed on system" ON)
    option(USE_SYSTEM_BLAS "Use BLAS/LAPACK installed on system" OFF)
    option(USE_SYSTEM_EIGEN "Use EIGEN installed on system" ON)
endif()


find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

if("${USE_SYSTEM_EIGEN}")
    find_package(Eigen3 CONFIG REQUIRED)
endif()

if("${USE_SYSTEM_BLAS}")
    find_package(BLAS REQUIRED)
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

if(${BUILD_DOCS})
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
	set(DOXYGEN_IN "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in")
	set(DOXYGEN_OUT "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out")
	configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
	message(STATUS "Start building documentation with Doxygen")
	add_custom_target(docs
	    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	    COMMENT "Generating documentation with Doxygen"
	    VERBATIM)
    else(DOXYGEN_FOUND)
	message(STATUS "Doxygen must be installed to generate documentation")
    endif(DOXYGEN_FOUND)
endif()
