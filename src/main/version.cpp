#include <Eigen/Core>
#include <fmt/core.h>
#include <gemmi/version.hpp>
#include <occ/3rdparty/cint_wrapper.h>
#include <occ/core/log.h>
#include <occ/main/version.h>
#include <xc.h>

namespace occ::main {

void print_header() {
    const std::string xc_version_string{XC_VERSION};
    const auto eigen_version_string =
        fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
                    EIGEN_MINOR_VERSION);
    const std::string cint_version_string{libcint::cint_version_string};
    const std::string gemmi_version_string{GEMMI_VERSION};
    const int fmt_major = FMT_VERSION / 10000;
    const int fmt_minor = (FMT_VERSION % 10000) / 100;
    const int fmt_patch = (FMT_VERSION % 100);
    const std::string fmt_version_string =
        fmt::format("{}.{}.{}", fmt_major, fmt_minor, fmt_patch);
    const std::string spdlog_version_string = fmt::format(
        "{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);

    log::info(R"(
   
	 Open
	  \
	   Comp
	    \
	     Chem

a library and program for quantum chemistry

copyright (C) 2020-2022 Peter Spackman 

this version of occ makes use of the following third party libraries:

CLI11                command line argument parser
eigen3               Linear Algebra (v {})
fmt                  String formatting (v {})
gau2grid             Gaussian basis function evaluation (v 2.0.7)
gemmi                CIF parsing & structure refinement (v {})
LBFGS++              L-BFGS implementation
libcint              Electron integrals using GTOs (v {})
libxc                Density functional implementations (v {})
nanoflann	     KDtree implementation
nlohmann::json       JSON parser
phmap                Fast hashmap implementation
pocketFFT 	     Standalone implementation of fast Fourier transform
scnlib               String parsing
spdlog               Logging (v {})
subprocess           Calling external subprocesses

)",
              eigen_version_string, fmt_version_string, gemmi_version_string,
              cint_version_string, xc_version_string, spdlog_version_string);
}
} // namespace occ::main
