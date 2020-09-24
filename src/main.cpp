#include "cxxopts.hpp"
#include "hf.h"
#include "molecule.h"
#include "scf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include "numgrid.h"

int main(int argc, const char **argv) {
  using craso::chem::Molecule;
  using craso::hf::HartreeFock;
  using craso::scf::RestrictedSCF;
  using craso::scf::UnrestrictedSCF;
  using std::cerr;
  using std::cout;
  using std::endl;

  cxxopts::Options options(
      "craso", "A quantum chemistry program for molecular crystals");
  options.add_options()
      ("b,basis", "Basis set name", cxxopts::value<std::string>()->default_value("3-21G"))
      ("j,nthreads", "Number of threads", cxxopts::value<int>()->default_value("1"))
      ("i,input", "Input file geometry", cxxopts::value<std::string>())
      ("h,help", "Print this help message")("uhf", "Use unrestricted")
      ("multiplicity", "Set the multiplicity of the system", cxxopts::value<int>()->default_value("1"))
      ("c,charge", "Set the total system charge", cxxopts::value<int>()->default_value("0"))
      ("diis-iteration", "Set the total system charge", cxxopts::value<int>()->default_value("2"));

  auto result = options.parse(argc, argv);
  if (result.count("help") || !(result.count("input"))) {
    fmt::print("{}\n", options.help());
    return 0;
  }

  try {
    libint2::Shell::do_enforce_unit_normalization(false);
    if (!libint2::initialized())
      libint2::initialize();
    const auto filename = result["input"].as<std::string>();
    const auto basisname = result["basis"].as<std::string>();
    const auto multiplicity = result["multiplicity"].as<int>();
    const auto charge = result["charge"].as<int>();
    const auto diis_iter = result["diis-iteration"].as<int>();
    Molecule m = craso::chem::read_xyz_file(filename);

    fmt::print("Geometry\n");
    for (const auto &atom : m.atoms()) {
      fmt::print("{:3d} {:10.6f} {:10.6f} {:10.6f}\n", atom.atomic_number,
                 atom.x, atom.y, atom.z);
    }
    bool unrestricted = result.count("uhf") || (multiplicity != 1);

    using craso::parallel::nthreads;
    nthreads = result["nthreads"].as<int>();
    omp_set_num_threads(nthreads);
    fmt::print("Using {} threads\n", nthreads);
    fmt::print("Geometry loaded from {}\n", filename);
    fmt::print("Using {} basis on all atoms\n", basisname);

    libint2::BasisSet obs(basisname, m.atoms());

    fmt::print("Orbital basis set rank = {}\n", obs.nbf());

    HartreeFock hf(m.atoms(), obs);
    if (unrestricted) {
      UnrestrictedSCF<HartreeFock> scf(hf, diis_iter);
      scf.set_multiplicity(multiplicity);
      scf.set_charge(charge);
      fmt::print("Multiplicity: {}\n", scf.multiplicity());
      fmt::print("n_alpha: {}\n", scf.n_alpha);
      fmt::print("n_beta: {}\n", scf.n_beta);
      double e = scf.compute_scf_energy();
      scf.print_orbital_energies();
    } else {
      RestrictedSCF<HartreeFock> scf(hf, diis_iter);
      double e = scf.compute_scf_energy();
      scf.print_orbital_energies();
    }

  } catch (const char *ex) {
    fmt::print("Caught exception when performing HF calculation:  {}\n", ex);
    return 1;
  } catch (std::string &ex) {
    fmt::print("Caught exception when performing HF calculation:  {}\n", ex);
    return 1;
  } catch (std::exception &ex) {
    fmt::print("Caught exception when performing HF calculation:  {}\n",
               ex.what());
    return 1;
  } catch (...) {
    fmt::print("Unknown exception occurred...\n");
    return 1;
  }
  return 0;
}
