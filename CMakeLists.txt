cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)

set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "5")
set(PROJECT_VERSION_PATCH "7")

project(occ
    VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}"
    LANGUAGES CXX C
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(
        FATAL_ERROR
        "In-source builds are not supported. Please make a new directory (i.e. build directory) and run cmake from there."
    )
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(SKBUILD)
  set(WITH_PYTHON_BINDINGS ON)
endif()


if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    option(BUILD_FPIC "Build with position independent code" OFF)
    option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
    option(WITH_BENCHMARKS "Enable benchmarking tests" OFF)
    option(WITH_CLANG_TIDY "Enable clang tidy" OFF)
    option(WITH_PYTHON_BINDINGS "Build python bindings" OFF)
    option(WITH_TBLITE "Include the tblite API in the build" OFF)
    option(BUILD_DOCS "Build documentation using Doxygen" OFF)
    option(USE_MLX "Use Apple MLX library for GPU offloading (WIP)" OFF)
    option(USE_SYSTEM_LIBXC "Use LibXC installed on system" OFF)
    option(USE_SYSTEM_BLAS "Use BLAS/LAPACK installed on system" OFF)
    option(USE_SYSTEM_EIGEN "Use EIGEN installed on system" OFF)
    option(USE_QCINT "Use SIMD optimised qcint rather than libcint" OFF)
    option(USE_OPENMP "Use OpenMP for threading rather than std::thread" ON)
    option(GG_NO_PRAGMA "Define the __GG_NO_PRAGMA macro" OFF)
    option(EMSCRIPTEN "Using emscripten" OFF)
    option(ENABLE_HOST_OPT "Enable -march=native and similar flags" ON)
    if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
      set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
endif()


if("${WITH_PYTHON_BINDINGS}")
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module
	OPTIONAL_COMPONENTS Development.SABIModule)
    message(STATUS "Enabled python bindings, setting CMAKE_POSITION_INDEPENDENT_CODE globally")
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()



if("${USE_OPENMP}")
find_package(OpenMP)
if(NOT "${OpenMP_FOUND}")
  message(STATUS "Could not find OpenMP, building without")
  set(USE_OPENMP OFF)
endif()
endif()

find_package(Threads REQUIRED)

if("${USE_SYSTEM_EIGEN}")
    find_package(Eigen3 CONFIG REQUIRED)
endif()

if("${USE_SYSTEM_BLAS}")
    find_package(BLAS REQUIRED)
endif()

if(${EMSCRIPTEN})
message(STATUS "Emscripten build flags")
# Set compiler and linker flags for pthreads
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -s USE_PTHREADS=1")
#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread -s USE_PTHREADS=1")

# Set Emscripten-specific options
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s NO_EXIT_RUNTIME=1")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s EXPORTED_RUNTIME_METHODS=['callMain']")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --pre-js ${CMAKE_CURRENT_SOURCE_DIR}/js/pre.js")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s STACK_SIZE=5MB")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s USE_PTHREADS=1 -pthread -s SHARED_MEMORY=1")
message(STATUS "${CMAKE_EXE_LINKER_FLAGS}")
endif()


include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake")
# don't use install commands from third party
add_subdirectory(3rdparty EXCLUDE_FROM_ALL)
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

include(GNUInstallDirs)
include(InstallRequiredSystemLibraries)
install(TARGETS occ
    EXPORT occ-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY share/basis DESTINATION "share/occ")
install(DIRECTORY share/solvent DESTINATION "share/occ")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixBundle.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/FixBundle.cmake"
    @ONLY
)

include(Packing)
