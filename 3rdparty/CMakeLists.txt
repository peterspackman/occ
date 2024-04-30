CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    VERSION "10.2.0"
    GIT_TAG "10.2.0"
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.12.0"
)

CPMAddPackage(
    NAME unordered_dense
    GITHUB_REPOSITORY "martinus/unordered_dense"
    VERSION "4.1.2"
)

CPMAddPackage(
    NAME scnlib
    GITHUB_REPOSITORY "eliaskosunen/scnlib"
    VERSION "2.0.2"
    OPTIONS "SIMDUTF_TESTS OFF" "ENABLE_FULL OFF"
)


CPMAddPackage(
    NAME CLI11
    GITHUB_REPOSITORY "CLIUtils/CLI11"
    VERSION "2.4.1"
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
    VERSION "6.1.2"
)
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
    GIT_TAG "v1.6.2"
)
endif()
