#include "molecule.h"
#include "hf.h"
#include "scf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "cxxopts.hpp"
#include <iostream>

int main(int argc, const char** argv) {
    using std::cout;
    using std::cerr;
    using std::endl;
    using craso::chem::Molecule;
    using craso::hf::HartreeFock;
    using craso::scf::SCF;

    cxxopts::Options options("craso", "A quantum chemistry program for molecular crystals");
    options.add_options()
        ("b,basis", "Basis set name", cxxopts::value<std::string>()->default_value("3-21G"))
        ("j,nthreads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        ("i,input", "Input file geometry", cxxopts::value<std::string>())
        ("h,help", "Print this help message")
        ("uhf", "Use unrestricted")
        ("multiplicity", "Set the multiplicity of the system", cxxopts::value<int>()->default_value("1"))
        ("c,charge", "Set the total system charge", cxxopts::value<int>()->default_value("0"))
    ;
    
    auto result = options.parse(argc, argv);
    if(result.count("help") || !(result.count("input")))
    {
        fmt::print("{}\n", options.help());
        return 0;
    }


    try {
        libint2::Shell::do_enforce_unit_normalization(false);
        if (!libint2::initialized()) libint2::initialize();
        const auto filename = result["input"].as<std::string>();
        const auto basisname = result["basis"].as<std::string>();
        const auto multiplicity = result["multiplicity"].as<int>();
        const auto charge = result["charge"].as<int>();
        Molecule m = craso::chem::read_xyz_file(filename);
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
        craso::scf::SCFKind scf_kind = craso::scf::SCFKind::rhf;
        if(unrestricted) scf_kind = craso::scf::SCFKind::uhf;
        craso::scf::SCF<HartreeFock> scf(hf, scf_kind);
        scf.set_multiplicity(multiplicity);
        scf.set_charge(charge);
        fmt::print("Multiplicity: {}\n", scf.multiplicity());
        fmt::print("n_alpha: {}\n", scf.n_alpha);
        fmt::print("n_beta: {}\n", scf.n_beta);
        double e = scf.compute_scf_energy();
        fmt::print("Total Energy (SCF): {:20.12f} hartree\n", e);
        fmt::print("Orbital energies alpha:\n{}\n", scf.orbital_energies_alpha);
        fmt::print("Orbital energies beta:\n{}\n", scf.orbital_energies_beta);
        if (false) {
            fmt::print("Density Matrix (alpha)\n{}\n", scf.Da);
            fmt::print("Density Matrix (beta)\n{}\n", scf.Db);
            fmt::print("Fock Matrix (alpha)\n{}\n", scf.Fa);
            fmt::print("Fock Matrix (beta)\n{}\n", scf.Fb);
        }

    }
    catch (const char* ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex);
        return 1;
    } catch (std::string& ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex);
        return 1;
    } catch (std::exception& ex) {
        fmt::print("Caught exception when performing HF calculation:  {}\n",  ex.what());
        return 1;
    } catch (...) {
        fmt::print("Unknown exception occurred...\n");
        return 1;
    }
    return 0;
}
