#include "cxxopts.hpp"
#include "hf.h"
#include "molecule.h"
#include "scf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include "numgrid.h"

void print_header()
{
    fmt::print("----------------------------------------------------\n");
    fmt::print("  craso - quantum chemistry and crystal structures  \n");
    fmt::print("\n\n     Copyright: Peter Spackman 2020\n\n");
    fmt::print("----------------------------------------------------\n");

}


int main(int argc, const char **argv) {
  using craso::chem::Molecule;
  using craso::chem::Element;
  using craso::hf::HartreeFock;
  using craso::scf::RestrictedSCF;
  using craso::scf::UnrestrictedSCF;
  using craso::scf::GeneralSCF;
  using std::cerr;
  using std::cout;
  using std::endl;

  cxxopts::Options options(
      "craso", "A quantum chemistry program for molecular crystals");
  options.add_options()
      ("b,basis", "Basis set name", cxxopts::value<std::string>()->default_value("3-21G"))
      ("j,nthreads", "Number of threads", cxxopts::value<int>()->default_value("1"))
      ("i,input", "Input file geometry", cxxopts::value<std::string>())
      ("h,help", "Print this help message")
      ("ghf", "Use general")
      ("uhf", "Use unrestricted")
      ("multiplicity", "Set the multiplicity of the system", cxxopts::value<int>()->default_value("1"))
      ("c,charge", "Set the total system charge", cxxopts::value<int>()->default_value("0"));

  auto result = options.parse(argc, argv);
  if (result.count("help") || !(result.count("input"))) {
    fmt::print("{}\n", options.help());
    return 0;
  }
  print_header();

  try {
    libint2::Shell::do_enforce_unit_normalization(false);
    if (!libint2::initialized())
      libint2::initialize();
    const auto filename = result["input"].as<std::string>();
    const auto basisname = result["basis"].as<std::string>();
    const auto multiplicity = result["multiplicity"].as<int>();
    const auto charge = result["charge"].as<int>();
    Molecule m = craso::chem::read_xyz_file(filename);

    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", filename, "el", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
      fmt::print("{:3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                 atom.x, atom.y, atom.z);
    }
    bool unrestricted = result.count("uhf") || (multiplicity != 1);
    bool general = result.count("ghf");

    using craso::parallel::nthreads;
    nthreads = result["nthreads"].as<int>();
    omp_set_num_threads(nthreads);
    fmt::print("{:12s} {:>12d}\n", "threads", nthreads);
    fmt::print("{:12s} {:>12d}\n", "eigen", Eigen::nbThreads());
    fmt::print("{:12s} {:>12s}\n", "basis", basisname);

    libint2::BasisSet obs(basisname, m.atoms());

    fmt::print("{:12s} {:>12d}\n", "n_bf", obs.nbf());

    HartreeFock hf(m.atoms(), obs);
    if (general) {
      fmt::print("{:12s} {:>12s}\n", "procedure", "ghf");
      GeneralSCF<HartreeFock> scf(hf);
      scf.conv = 1e-12;
      scf.set_charge(charge);
      double e = scf.compute_scf_energy();
    } else if (unrestricted) {
      fmt::print("{:12s} {:>12s}\n", "procedure", "uhf");
      UnrestrictedSCF<HartreeFock> scf(hf);
      scf.conv = 1e-12;
      scf.set_multiplicity(multiplicity);
      scf.set_charge(charge);
      double e = scf.compute_scf_energy();
      //scf.print_orbital_energies();

    } else
    {
      fmt::print("{:12s} {:>12s}\n", "procedure", "rhf");
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
