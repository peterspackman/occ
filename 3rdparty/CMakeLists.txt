CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    VERSION "11.1.0"
    GIT_TAG "11.1.0"
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.x"
)

if(NOT "${USE_SYSTEM_TBB}")

set(TBB_OPTIONS "")
list(APPEND TBB_OPTIONS "TBB_TEST OFF")
list(APPEND TBB_OPTIONS "TBB_STRICT OFF")
# Enable HWLOC for better NUMA topology detection on native builds, disable for Emscripten
if(EMSCRIPTEN)
    list(APPEND TBB_OPTIONS "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH ON")
else()
    list(APPEND TBB_OPTIONS "TBB_DISABLE_HWLOC_AUTOMATIC_SEARCH OFF")
endif()
list(APPEND TBB_OPTIONS "BUILD_SHARED_LIBS OFF")

# Windows MinGW-specific workarounds for TBB compilation issues
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND TBB_OPTIONS "CMAKE_CXX_FLAGS=-O2 -fno-stack-clash-protection -fno-fcf-protection")
endif()

CPMAddPackage(
    NAME oneTBB
    GITHUB_REPOSITORY "uxlfoundation/oneTBB"
    VERSION "2022.2.0"
    OPTIONS ${TBB_OPTIONS}
)
endif()

CPMAddPackage(
    NAME unordered_dense
    GITHUB_REPOSITORY "martinus/unordered_dense"
    VERSION "4.5.0"
)

CPMAddPackage(
    NAME scnlib
    GITHUB_REPOSITORY "eliaskosunen/scnlib"
    VERSION "4.0.1"
    OPTIONS "SIMDUTF_TESTS OFF" "ENABLE_FULL OFF"
)


CPMAddPackage(
    NAME CLI11
    GITHUB_REPOSITORY "CLIUtils/CLI11"
    VERSION "2.4.2"
)

if("${USE_MLX}")
CPMAddPackage(
    NAME mlx
    GITHUB_REPOSITORY "ml-explore/mlx"
    VERSION "0.6.0"
)
else()
CPMAddPackage(
    NAME "nlohmann_json"
    GITHUB_REPOSITORY "nlohmann/json"
    VERSION "3.11.3"
)

endif()

if(NOT "${USE_SYSTEM_EIGEN}")
CPMAddPackage(
    NAME eigen3
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
    DOWNLOAD_ONLY YES  # Eigen CMakelists.txt not intended for library use
)

# Add the Eigen library target to link against
if("${eigen3_ADDED}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen3_SOURCE_DIR}")
endif()

endif()

CPMAddPackage(
    NAME gemmi
    GITHUB_REPOSITORY project-gemmi/gemmi
    VERSION "0.6.5"
    DOWNLOAD_ONLY YES
)

# Add Gemmi as a header only library for linking
if("${gemmi_ADDED}")
    add_library(gemmi::gemmi INTERFACE IMPORTED GLOBAL)
    target_include_directories(gemmi::gemmi INTERFACE "${gemmi_SOURCE_DIR}/include")
endif()

CPMAddPackage(
    NAME dftd4_cpp
    GITHUB_REPOSITORY "peterspackman/cpp-d4"
    GIT_TAG main
    VERSION "2.2.0"
    OPTIONS "DFTD4_USE_EIGEN ON" "BUILD_SHARED_LIBS OFF"
)

CPMAddPackage(
    NAME LBFGSpp
    GITHUB_REPOSITORY yixuan/LBFGSpp
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

# Add LBFGSpp as a header only library for linking
if("${LBFGSpp_ADDED}")
    add_library(LBFGSpp::LBFGSpp INTERFACE IMPORTED GLOBAL)
    target_include_directories(LBFGSpp::LBFGSpp INTERFACE "${LBFGSpp_SOURCE_DIR}/include")
endif()

if("${USE_QCINT}")
    set(LIBCINT_REPOSITORY "sunqm/qcint")
else()
    set(LIBCINT_REPOSITORY "peterspackman/libcint")
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(LIBCINT_BUILD_TYPE "CMAKE_BUILD_TYPE Release")
endif ()

# Add compiler flags to fix the implicit function declaration error
set(LIBCINT_C_FLAGS "-Wno-implicit-function-declaration -Wno-deprecated-non-prototype -D_GNU_SOURCE")

CPMAddPackage(
    NAME libcint
    GITHUB_REPOSITORY "${LIBCINT_REPOSITORY}"
    GIT_TAG master
    OPTIONS
        "WITH_FORTRAN OFF"
        "WITH_CINT2_INTERFACE OFF"
        "ENABLE_STATIC ON"
        "BUILD_SHARED_LIBS OFF"
        "PYPZPX ON"
        "BUILD_MARCH_NATIVE ON"
        "WITH_RANGE_COULOMB ON"
        "CMAKE_C_FLAGS ${LIBCINT_C_FLAGS}"
        ${LIBCINT_BUILD_TYPE}
    VERSION "6.1.2"
)

# Apply the flags to the target after it's created
if(TARGET cint)
    target_compile_options(cint PRIVATE 
        -Wno-implicit-function-declaration
        -Wno-deprecated-non-prototype
    )
endif()

add_library(libcint::libcint INTERFACE IMPORTED GLOBAL) 
target_link_libraries(libcint::libcint INTERFACE cint)
target_include_directories(
    libcint::libcint INTERFACE
    "${libcint_SOURCE_DIR}/include"
    "${libcint_SOURCE_DIR}/src"
    "${libcint_BINARY_DIR}/include"
)

if(NOT "${USE_SYSTEM_LIBXC}")
CPMAddPackage(
    NAME Libxc
    URL "https://gitlab.com/libxc/libxc/-/archive/6.2.2/libxc-6.2.2.tar.gz"
    OPTIONS
        "BUILD_TESTING OFF"
        "ENABLE_XHOST ${ENABLE_HOST_OPT}"
        "BUILD_FPIC ${WITH_PYTHON_BINDINGS}"
        "CMAKE_POLICY_VERSION_MINIMUM 3.5"
)
add_library(Libxc::xc INTERFACE IMPORTED GLOBAL) 
target_link_libraries(Libxc::xc INTERFACE xc)
target_include_directories(
    Libxc::xc INTERFACE
    "${Libxc_SOURCE_DIR}/src"
    "${Libxc_BINARY_DIR}/src"
    "${Libxc_BINARY_DIR}"
)

endif()


if("${WITH_PYTHON_BINDINGS}")
CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY "wjakob/nanobind"
    GIT_TAG "v2.4.0"
)
endif()
