#include <occ/main/version.h>
#include <boost/version.hpp>
#include <gemmi/version.hpp>
#include <fmt/core.h>
#include <Eigen/Core>
#include <libint2.hpp>
#include <xc.h>
#include <spdlog/spdlog.h>


namespace occ::main {

void print_header() {
    const std::string xc_version_string{XC_VERSION};
    const auto eigen_version_string =
        fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
                    EIGEN_MINOR_VERSION);
    const std::string libint_version_string{LIBINT_VERSION};
    const std::string gemmi_version_string{GEMMI_VERSION};
    const std::string boost_version_string{BOOST_LIB_VERSION};
    const int fmt_major = FMT_VERSION / 10000;
    const int fmt_minor = (FMT_VERSION % 10000) / 100;
    const int fmt_patch = (FMT_VERSION % 100);
    const std::string fmt_version_string =
        fmt::format("{}.{}.{}", fmt_major, fmt_minor, fmt_patch);
    const std::string spdlog_version_string = fmt::format(
        "{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);

    fmt::print(R"(
   
	 Open
	  \
	   Comp
	    \
	     Chem

a library and program for quantum chemistry

copyright (C) 2020-2022 Peter Spackman 

this version of occ makes use of the following third party libraries:

eigen3               Linear Algebra (v {})
libint2              Electron integrals using GTOs (v {})
libxc                Density functional implementations (v {})
gau2grid             Gaussian basis function evaluation (v 2.0.7)
gemmi                CIF parsing & structure refinement (v {})
boost::graph         Graph implementation (v {})
fmt                  String formatting (v {})
spdlog               Logging (v {})
scnlib               String parsing
nlohmann::json       JSON parser
cxxopts              command line argument parser
LBFGS++              L-BFGS implementation
zlib                 zip library

)",
               eigen_version_string, libint_version_string, xc_version_string,
               gemmi_version_string, boost_version_string, fmt_version_string,
               spdlog_version_string);
}
}
