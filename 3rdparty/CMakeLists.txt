CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    VERSION "9.1.0"
    GIT_TAG "9.1.0"
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.10.0"
)

CPMAddPackage(
    NAME unordered_dense
    GITHUB_REPOSITORY "martinus/unordered_dense"
    VERSION "4.0.4"
)

CPMAddPackage(
    NAME scnlib
    GITHUB_REPOSITORY "peterspackman/scnlib"
    GIT_TAG "master"
)

CPMAddPackage(
    NAME "nlohmann_json"
    GITHUB_REPOSITORY "nlohmann/json"
    VERSION "3.10.4"
)

CPMAddPackage(
    NAME CLI11
    GITHUB_REPOSITORY "CLIUtils/CLI11"
    GIT_TAG "1a26afab049bb75c0523a754d62b961439248d44"
)

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
    GIT_TAG master
    DOWNLOAD_ONLY YES
)

# Add Gemmi as a header only library for linking
if("${gemmi_ADDED}")
    add_library(gemmi::gemmi INTERFACE IMPORTED GLOBAL)
    target_include_directories(gemmi::gemmi INTERFACE "${gemmi_SOURCE_DIR}/include")
endif()


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
    set(LIBCINT_REPOSITORY "sunqm/libcint")
endif()

CPMAddPackage(
    NAME libcint
    GITHUB_REPOSITORY "${LIBCINT_REPOSITORY}"
    GIT_TAG master
    OPTIONS "WITH_FORTRAN OFF" "WITH_CINT2_INTERFACE OFF" "ENABLE_STATIC ON" "BUILD_SHARED_LIBS OFF" "PYPZPX ON" "BUILD_MARCH_NATIVE ON" "WITH_RANGE_COULOMB ON"
    VERSION "5.4.0"
)
add_library(libcint::libcint INTERFACE IMPORTED GLOBAL) 
target_link_libraries(libcint::libcint INTERFACE cint)
target_include_directories(
    libcint::libcint INTERFACE
    "${libcint_SOURCE_DIR}/include"
    "${libcint_SOURCE_DIR}/src"
    "${libcint_BINARY_DIR}/include"
)

CPMAddPackage(
    NAME libecpint
    GITHUB_REPOSITORY "peterspackman/libecpint"
    GIT_TAG "master"
    OPTIONS "LIBECPINT_BUILD_TESTS OFF" "LIBECPINT_BUILD_DOCS OFF"
)

if(NOT "${USE_SYSTEM_LIBXC}")
CPMAddPackage(
    NAME Libxc
    GIT_REPOSITORY "https://gitlab.com/libxc/libxc.git"
    GIT_TAG 5.2.3
    OPTIONS "BUILD_TESTING OFF" "ENABLE_XHOST ${ENABLE_HOST_OPT}" "BUILD_FPIC ${WITH_PYTHON_BINDINGS}"
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
    GIT_TAG "v1.1.1"
)
endif()
