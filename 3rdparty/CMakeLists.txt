
CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    GIT_TAG master
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.x"
)

CPMAddPackage(
    NAME scnlib
    GITHUB_REPOSITORY "eliaskosunen/scnlib"
    GIT_TAG master
)

CPMAddPackage(
    NAME gau2grid
    GITHUB_REPOSITORY "dgasmith/gau2grid"
    GIT_TAG master
)

CPMAddPackage(
    NAME toml11
    GITHUB_REPOSITORY "ToruNiina/toml11"
    GIT_TAG master
)

# the install option has to be explicitly set to allow installation
CPMAddPackage(
    NAME cxxopts
    GITHUB_REPOSITORY jarro2783/cxxopts
    VERSION 2.2.1
    OPTIONS "CXXOPTS_BUILD_EXAMPLES NO" "CXXOPTS_BUILD_TESTS NO" "CXXOPTS_ENABLE_INSTALL YES"
)

if(NOT "${USE_SYSTEM_EIGEN}")
CPMAddPackage(
    NAME eigen3
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.3.9/eigen-3.3.9.zip"
    DOWNLOAD_ONLY YES  # Eigen CMakelists.txt not intended for library use
)

# Add the Eigen library target to link against
if("${eigen3_ADDED}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
    target_include_directories(Eigen3::Eigen INTERFACE ${eigen3_SOURCE_DIR})
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

# Fastor currently unused
#CPMAddPackage(
#    NAME Fastor
#    GITHUB_REPOSITORY romeric/Fastor
#    GIT_TAG master
#    DOWNLOAD_ONLY YES
#)

## Add Fastor as a header only library for linking
#if("${Fastor_ADDED}")
#    add_library(Fastor::Fastor INTERFACE IMPORTED GLOBAL)
#    target_include_directories(Fastor::Fastor INTERFACE "${Fastor_SOURCE_DIR}")
#endif()

if(NOT "${USE_SYSTEM_ZLIB}")
CPMAddPackage(
    NAME zlib
    URL "https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.5.zip"
    OPTIONS "ZLIB_COMPAT ON" "ZLIB_ENABLE_TESTS OFF"
)
add_library(ZLIB::ZLIB ALIAS zlib)
endif()

# Need to get OpenBLAS working
#if(NOT "${USE_SYSTEM_BLAS}")
#CPMAddPackage(
#    NAME OpenBLAS
#    GITHUB_REPOSITORY xianyi/OpenBLAS
#    VERSION 0.3.16
#)
#endif()


if(NOT "${USE_SYSTEM_LIBINT2}")
set(LIBINT2_URL "https://github.com/evaleev/libint/releases/download/v2.7.0-beta.6/libint-2.7.0-beta.6.tgz")
CPMAddPackage(
    NAME Libint2
    URL "${LIBINT2_URL}"
)
add_library(Libint2::cxx ALIAS libint2_cxx)
endif()

if(NOT "${USE_SYSTEM_LIBXC}")
CPMAddPackage(
    NAME Libxc
    GIT_REPOSITORY "https://gitlab.com/libxc/libxc.git"
    GIT_TAG 5.0.0
    OPTIONS "BUILD_TESTING OFF"
)
add_library(Libxc::xc INTERFACE IMPORTED GLOBAL) 
target_link_libraries(Libxc::xc INTERFACE xc)
target_include_directories(
    Libxc::xc INTERFACE
    "${Libxc_BINARY_DIR}/gen_funcidx"
    "${Libxc_SOURCE_DIR}/src"
    "${Libxc_BINARY_DIR}/src"
    "${Libxc_BINARY_DIR}"
)
endif()

