#include <tonto/core/logger.h>
#include <tonto/core/molecule.h>
#include <tonto/core/timings.h>
#include <tonto/io/fchkwriter.h>
#include <tonto/qm/dft.h>
#include <tonto/qm/hf.h>
#include <tonto/qm/scf.h>
#include <tonto/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <xc.h>
#include "gemmi/version.hpp"
#include "boost/version.hpp"
#include <spdlog/cfg/env.h>

void print_header()
{
    const std::string xc_version_string{XC_VERSION};
    const auto eigen_version_string = fmt::format("{}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
    const std::string libint_version_string{LIBINT_VERSION};
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
libxc        - Density functional implementations (v {})
gemmi        - CIF parsing & structure refinement (v {})
boost::graph - Graph implementation (v {})
OpenMP       - Multithreading
fmt          - String formatting (v {})
spdlog       - Logging (v {})

)", eigen_version_string, libint_version_string, xc_version_string,
    gemmi_version_string, boost_version_string, fmt_version_string, spdlog_version_string);
}


int main(int argc, const char **argv) {
    using tonto::chem::Molecule;
    using tonto::chem::Element;
    using tonto::hf::HartreeFock;
    using tonto::scf::SCF;
    using tonto::qm::SpinorbitalKind;
    using tonto::qm::Wavefunction;
    using std::cerr;
    using std::cout;
    using std::endl;

    argparse::ArgumentParser parser("tonto");
    parser.add_argument("input").help("Input file geometry");
    parser.add_argument("-b", "--basis").help("Basis set name")
            .default_value(std::string("3-21G"));
    parser.add_argument("-j", "--threads")
            .help("Number of threads")
            .default_value(1)
            .action([](const std::string& value) { return std::stoi(value); });
    parser.add_argument("--method")
            .default_value(std::string("rhf"));

    parser.add_argument("-c", "--charge")
            .help("System charge")
            .default_value(0)
            .action([](const std::string& value) { return std::stoi(value); });

    parser.add_argument("-m", "--multiplicity")
            .help("System multiplicity")
            .default_value(1)
            .action([](const std::string& value) { return std::stoi(value); });

    parser.add_argument("--uks")
            .help("Use unrestricted dft")
            .default_value(false)
            .implicit_value(true);


    parser.add_argument("-df", "--df-basis").help("Density fitting basis name");
    tonto::timing::start(tonto::timing::category::global);
    tonto::log::set_level(tonto::log::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }
    spdlog::set_level(spdlog::level::info);

    print_header();

    try {
        libint2::Shell::do_enforce_unit_normalization(false);
        libint2::initialize();
        const auto filename = parser.get<std::string>("input");
        const auto basisname = parser.get<std::string>("--basis");
        const auto multiplicity = parser.get<int>("--multiplicity");
        const auto charge = parser.get<int>("--charge");
        Molecule m = tonto::chem::read_xyz_file(filename);

        fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", filename, "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                       atom.x, atom.y, atom.z);
        }
        const auto method = parser.get<std::string>("--method");
        using tonto::parallel::nthreads;
        nthreads = parser.get<int>("--threads");
        omp_set_num_threads(nthreads);
        fmt::print("\nParallelization: {} OpenMP threads, {} Eigen threads\n", nthreads, Eigen::nbThreads());
        fmt::print("Input method string: '{}'\n", method);
        fmt::print("Input basis name: '{}'\n", basisname);

        tonto::qm::BasisSet basis(basisname, m.atoms());
        basis.set_pure(false);
        fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
        Wavefunction wfn;
        if (method == "rhf") {
            HartreeFock hf(m.atoms(), basis);
            tonto::log::debug("Initializing restricted SCF");
            SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
            scf.set_charge(charge);
            if(auto dfbasis = parser.present("--df-basis")) {
                scf.set_density_fitting_basis(*dfbasis);
            }
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        } else if (method == "ghf") {
            HartreeFock hf(m.atoms(), basis);
            tonto::log::debug("Initializing general SCF");
            SCF<HartreeFock, SpinorbitalKind::General> scf(hf);
            scf.set_charge(charge);
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        } else if (method == "uhf") {
            HartreeFock hf(m.atoms(), basis);
            tonto::log::debug("Initializing unrestricted SCF");
            SCF<HartreeFock, SpinorbitalKind::Unrestricted> scf(hf);
            scf.set_charge_multiplicity(charge, multiplicity);
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        } else
        {
            if (parser.get<bool>("--uks")) {
                tonto::log::debug("Initializing unrestricted DFT");
                tonto::dft::DFT rks(method, basis, m.atoms(), SpinorbitalKind::Unrestricted);
                SCF<tonto::dft::DFT, SpinorbitalKind::Unrestricted> scf(rks);
                scf.set_charge_multiplicity(charge, multiplicity);
                scf.start_incremental_F_threshold = 0.0;
                double e = scf.compute_scf_energy();
                wfn = scf.wavefunction();
            }
            else {
                tonto::log::debug("Initializing restricted DFT");
                tonto::dft::DFT rks(method, basis, m.atoms(), SpinorbitalKind::Restricted);
                SCF<tonto::dft::DFT, SpinorbitalKind::Restricted> scf(rks);
                scf.set_charge_multiplicity(charge, multiplicity);
                scf.start_incremental_F_threshold = 0.0;
                double e = scf.compute_scf_energy();
                wfn = scf.wavefunction();
            }
        }
        tonto::io::FchkWriter fchk("test.fchk");
        fchk.set_title(filename);
        fchk.set_method(method);
        fchk.set_basis_name(basisname);
        wfn.save(fchk);
        fchk.write();

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
    tonto::timing::stop(tonto::timing::global);
    tonto::timing::print_timings();
    return 0;
}
