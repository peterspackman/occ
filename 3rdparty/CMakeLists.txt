if(NOT "${USE_SYSTEM_BOOST}")
CPMAddPackage(
  NAME Boost
  VERSION 1.76.0
  URL https://boostorg.jfrog.io/artifactory/main/release/1.76.0/source/boost_1_76_0.tar.gz
  DOWNLOAD_ONLY True 
)
if(Boost_ADDED)
  # Define the header-only Boost target
  add_library(Boost::boost INTERFACE IMPORTED GLOBAL)
  target_include_directories(Boost::boost SYSTEM INTERFACE ${Boost_SOURCE_DIR})

  # Disable autolink
  target_compile_definitions(Boost::boost INTERFACE BOOST_ALL_NO_LIB=1)
endif()
endif()

CPMAddPackage(
    NAME fmt
    GITHUB_REPOSITORY "fmtlib/fmt"
    GIT_TAG master
)

CPMAddPackage(
    NAME spdlog
    GITHUB_REPOSITORY "gabime/spdlog"
    OPTIONS "SPDLOG_FMT_EXTERNAL ON"
    VERSION "1.10.0"
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
    NAME nlohmann_json
    GITHUB_REPOSITORY nlohmann/json
    VERSION 3.10.4
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
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
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
set(LIBINT2_URL "https://github.com/evaleev/libint/releases/download/v2.7.1/libint-2.7.1.tgz")
CPMAddPackage(
    NAME Libint2
    URL "${LIBINT2_URL}"
    OPTIONS "LIBINT_HAS_SYSTEM_BOOST_PREPROCESSOR_VARIADICS ON"
)
add_library(Libint2::cxx ALIAS libint2_cxx)
endif()

if(NOT "${USE_SYSTEM_LIBXC}")
CPMAddPackage(
    NAME Libxc
    GIT_REPOSITORY "https://gitlab.com/libxc/libxc.git"
    GIT_TAG 5.2.3
    OPTIONS "BUILD_TESTING OFF"
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


