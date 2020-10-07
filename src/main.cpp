#include "logger.h"
#include "cxxopts.hpp"
#include "hf.h"
#include "molecule.h"
#include "scf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include "numgrid.h"
#include <xc.h>
#include "gemmi/version.hpp"
#include "boost/version.hpp"
#include "dft.h"

void print_header()
{
    const std::string xc_version_string{XC_VERSION};
    const auto eigen_version_string = fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    const std::string libint_version_string{LIBINT_VERSION};
    const std::string numgrid_verison_string{numgrid_get_version()};
    const std::string gemmi_version_string{GEMMI_VERSION};
    const std::string boost_version_string{BOOST_LIB_VERSION};
    const int fmt_major = FMT_VERSION / 10000;
    const int fmt_minor = (FMT_VERSION % 10000) / 100;
    const int fmt_patch = (FMT_VERSION % 100);
    const std::string fmt_version_string = fmt::format("{}.{}.{}", fmt_major, fmt_minor, fmt_patch);
    const std::string spdlog_version_string = fmt::format("{}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR, SPDLOG_VER_PATCH);

    fmt::print(R"(


              ,d                               ,d               
              88                               88               
            MM88MMM  ,adPPYba,   8b,dPPYba,  MM88MMM  ,adPPYba, 
              88    a8"     "8a  88P'   `"8a   88    a8"     "8a
              88    8b       d8  88       88   88    8b       d8
              88,   "8a,   ,a8"  88       88   88,   "8a,   ,a8"
              "Y888  `"YbbdP"'   88       88   "Y888  `"YbbdP"' 



    Copyright (C) 2020
    Peter Spackman - Primary Developer
    Dylan Jayatilaka 

This version of tonto also uses the following third party libraries:

    eigen        - Linear Algebra (v {})
    libint2      - Electron integrals using GTOs (v {})
    numgrid      - DFT grids (v {})
    libxc        - Density functional implementations (v {})
    gemmi        - CIF parsing & structure refinement (v {})
    boost::graph - Graph implementation (v {})
    OpenMP       - Multithreading
    fmt          - String formatting (v {})
    spdlog       - Logging (v {})

)", eigen_version_string, libint_version_string, numgrid_verison_string, xc_version_string,
    gemmi_version_string, boost_version_string, fmt_version_string, spdlog_version_string);
}


int main(int argc, const char **argv) {
  using tonto::chem::Molecule;
  using tonto::chem::Element;
  using tonto::hf::HartreeFock;
  using tonto::scf::RestrictedSCF;
  using tonto::scf::UnrestrictedSCF;
  using tonto::scf::GeneralSCF;
  using std::cerr;
  using std::cout;
  using std::endl;

  cxxopts::Options options(
      "tonto", "A quantum chemistry program for molecular crystals");
  options.add_options()
      ("b,basis", "Basis set name", cxxopts::value<std::string>()->default_value("3-21G"))
      ("j,nthreads", "Number of threads", cxxopts::value<int>()->default_value("1"))
      ("i,input", "Input file geometry", cxxopts::value<std::string>())
      ("h,help", "Print this help message")
      ("ghf", "Use general")
      ("uhf", "Use unrestricted")
      ("dft", "Use DFT")
      ("multiplicity", "Set the multiplicity of the system", cxxopts::value<int>()->default_value("1"))
      ("c,charge", "Set the total system charge", cxxopts::value<int>()->default_value("0"));

  tonto::log::set_level(tonto::log::level::debug);
  auto result = options.parse(argc, argv);
  if (result.count("help") || !(result.count("input"))) {
    fmt::print("{}\n", options.help());
    return 0;
  }
  print_header();

  try {
    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();
    const auto filename = result["input"].as<std::string>();
    const auto basisname = result["basis"].as<std::string>();
    const auto multiplicity = result["multiplicity"].as<int>();
    const auto charge = result["charge"].as<int>();
    Molecule m = tonto::chem::read_xyz_file(filename);

    fmt::print("Input geometry ({})\n    {:3s} {:^10s} {:^10s} {:^10s}\n", filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
      fmt::print("    {:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                 atom.x, atom.y, atom.z);
    }
    bool unrestricted = result.count("uhf") || (multiplicity != 1);
    bool general = result.count("ghf");
    bool dft = result.count("dft");

    using tonto::parallel::nthreads;
    nthreads = result["nthreads"].as<int>();
    omp_set_num_threads(nthreads);
    fmt::print("\n    {:12s} {:>12d}\n", "threads", nthreads);
    fmt::print("    {:12s} {:>12d}\n", "eigen", Eigen::nbThreads());
    fmt::print("    {:12s} {:>12s}\n", "basis", basisname);

    libint2::BasisSet obs(basisname, m.atoms());
    obs.set_pure(false);
    fmt::print("    {:12s} {:>12d}\n", "n_bf", obs.nbf());
    HartreeFock hf(m.atoms(), obs);
    if (dft) {
      fmt::print("    {:12s} {:>12s}\n", "procedure", "rks");
      tonto::dft::DFT rks("LDA", obs, m.atoms());
      RestrictedSCF<tonto::dft::DFT> scf(rks);
      double e = scf.compute_scf_energy();
    } else if (general) {
      fmt::print("    {:12s} {:>12s}\n", "procedure", "ghf");
      GeneralSCF<HartreeFock> scf(hf);
      scf.conv = 1e-12;
      scf.set_charge(charge);
      double e = scf.compute_scf_energy();
    } else if (unrestricted) {
      fmt::print("    {:12s} {:>12s}\n", "procedure", "uhf");
      UnrestrictedSCF<HartreeFock> scf(hf);
      scf.conv = 1e-12;
      scf.set_multiplicity(multiplicity);
      scf.set_charge(charge);
      double e = scf.compute_scf_energy();
      //scf.print_orbital_energies();
    } else
    {
      fmt::print("    {:12s} {:>12s}\n", "procedure", "rhf");
      RestrictedSCF<HartreeFock> scf(hf);
      scf.conv = 1e-12;
      scf.set_charge(charge);
      double e = scf.compute_scf_energy();
      //scf.print_orbital_energies();
    }

  } catch (const char *ex) {
    fmt::print("Caught exception when performing HF calculation:\n**{}**\n", ex);
    return 1;
  } catch (std::string &ex) {
    fmt::print("Caught exception when performing HF calculation:\n**{}**\n", ex);
    return 1;
  } catch (std::exception &ex) {
    fmt::print("Caught exception when performing HF calculation:\n**{}**\n",
               ex.what());
    return 1;
  } catch (...) {
    fmt::print("Unknown exception occurred...\n");
    return 1;
  }
  return 0;
}
