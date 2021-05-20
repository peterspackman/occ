#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/inputfile.h>
#include <occ/dft/dft.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <xc.h>
#include "gemmi/version.hpp"
#include "boost/version.hpp"
#include <spdlog/cfg/env.h>
#include <filesystem>

namespace fs = std::filesystem;
using occ::chem::Molecule;
using occ::chem::Element;
using occ::hf::HartreeFock;
using occ::scf::SCF;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

struct InputConfiguration {
    fs::path input_file;
    std::string method;
    std::string basis_name;
    std::optional<std::string> density_fitting_basis{std::nullopt};
    size_t multiplicity;
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int charge;
};

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

Open Computational Chemistry (OCC)

Copyright (C) 2020-2021
Peter Spackman - Primary Developer

This version of occ makse use of the following third party libraries:

eigen3       - Linear Algebra (v {})
libint2      - Electron integrals using GTOs (v {})
libxc        - Density functional implementations (v {})
gemmi        - CIF parsing & structure refinement (v {})
boost::graph - Graph implementation (v {})
fmt          - String formatting (v {})
spdlog       - Logging (v {})

)", eigen_version_string, libint_version_string, xc_version_string,
    gemmi_version_string, boost_version_string, fmt_version_string, spdlog_version_string);
}



Wavefunction run_from_xyz_file(const InputConfiguration &config)
{
    std::string filename = config.input_file.string();
    Molecule m = occ::chem::read_xyz_file(filename);

    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    fmt::print("Input method string: '{}'\n", config.method);
    fmt::print("Input basis name: '{}'\n", config.basis_name);

    occ::qm::BasisSet basis(config.basis_name, m.atoms());
    basis.set_pure(false);
    fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
    Wavefunction wfn;
    if (config.method == "rhf") {
        HartreeFock hf(m.atoms(), basis);
        occ::log::debug("Initializing restricted SCF");
        SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
        scf.set_charge(config.charge);
        if(config.density_fitting_basis) scf.set_density_fitting_basis(*config.density_fitting_basis);
        double e = scf.compute_scf_energy();
        wfn = scf.wavefunction();
    } else if (config.method == "ghf") {
        HartreeFock hf(m.atoms(), basis);
        occ::log::debug("Initializing general SCF");
        SCF<HartreeFock, SpinorbitalKind::General> scf(hf);
        scf.set_charge(config.charge);
        double e = scf.compute_scf_energy();
        wfn = scf.wavefunction();
    } else if (config.method == "uhf") {
        HartreeFock hf(m.atoms(), basis);
        occ::log::debug("Initializing unrestricted SCF");
        SCF<HartreeFock, SpinorbitalKind::Unrestricted> scf(hf);
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        double e = scf.compute_scf_energy();
        wfn = scf.wavefunction();
    } else
    {
        if (config.spinorbital_kind == SpinorbitalKind::Unrestricted) {
            occ::log::debug("Initializing unrestricted DFT");
            occ::dft::DFT rks(config.method, basis, m.atoms(), SpinorbitalKind::Unrestricted);
            SCF<occ::dft::DFT, SpinorbitalKind::Unrestricted> scf(rks);
            scf.set_charge_multiplicity(config.charge, config.multiplicity);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
        else {
            occ::log::debug("Initializing restricted DFT");
            occ::dft::DFT rks(config.method, basis, m.atoms(), SpinorbitalKind::Restricted);
            SCF<occ::dft::DFT, SpinorbitalKind::Restricted> scf(rks);
            scf.set_charge_multiplicity(config.charge, config.multiplicity);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
    }
    return wfn;
}

Wavefunction run_from_gaussian_input_file(const InputConfiguration &config)
{
    Wavefunction wfn;
    occ::io::GaussianInputFile com(config.input_file.string());
    Molecule m(com.atomic_numbers, com.atomic_positions);

    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", config.input_file, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    fmt::print("Input method string: '{}'\n", com.method);
    fmt::print("Input basis name: '{}'\n", com.basis_name);
    fmt::print("Total charge: {}\n", com.charge);
    fmt::print("System multiplicity: {}\n", com.multiplicity);

    occ::qm::BasisSet basis(com.basis_name, m.atoms());
    basis.set_pure(false);


    if(com.method_type == occ::io::GaussianInputFile::MethodType::HF)
    {
        if(com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
        {
            HartreeFock hf(m.atoms(), basis);
            occ::log::debug("Initializing unrestricted SCF");
            SCF<HartreeFock, SpinorbitalKind::Unrestricted> scf(hf);
            scf.set_charge_multiplicity(com.charge, com.multiplicity);
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
        else
        {
            HartreeFock hf(m.atoms(), basis);
            occ::log::debug("Initializing unrestricted SCF");
            SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
            scf.set_charge(com.charge);
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
    }
    else {
        if(com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
        {
            occ::log::debug("Initializing unrestricted DFT");
            occ::dft::DFT uks(com.method, basis, m.atoms(), SpinorbitalKind::Unrestricted);
            SCF<occ::dft::DFT, SpinorbitalKind::Unrestricted> scf(uks);
            scf.set_charge_multiplicity(com.charge, com.multiplicity);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
        else
        {
            occ::log::debug("Initializing unrestricted DFT");
            occ::dft::DFT rks(com.method, basis, m.atoms(), SpinorbitalKind::Restricted);
            SCF<occ::dft::DFT, SpinorbitalKind::Restricted> scf(rks);
            scf.set_charge(com.charge);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            wfn = scf.wavefunction();
        }
    }

    return wfn;
}


int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("occ");
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
    occ::timing::start(occ::timing::category::global);
    occ::log::set_level(occ::log::level::warn);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        occ::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }
    spdlog::set_level(spdlog::level::warn);

    print_header();


    const std::string error_format = "Exception:\n    {}\nTerminating program.\n";
    try {
        libint2::Shell::do_enforce_unit_normalization(false);
        libint2::initialize();
        InputConfiguration config;
        config.input_file = parser.get<std::string>("input");
        config.basis_name = parser.get<std::string>("--basis");
        config.multiplicity = parser.get<int>("--multiplicity");
        config.method = parser.get<std::string>("--method");
        config.charge = parser.get<int>("--charge");
        if (parser.get<bool>("--uks") || config.method == "uhf") {
            config.spinorbital_kind == SpinorbitalKind::Unrestricted;
        }
        else if(config.method == "ghf") config.spinorbital_kind == SpinorbitalKind::General;

        if(auto dfbasis = parser.present("--df-basis")) {
            config.density_fitting_basis = *dfbasis;
        }
        using occ::parallel::nthreads;
        nthreads = parser.get<int>("--threads");
        fmt::print("\nParallelization: {} threads, {} Eigen threads\n", nthreads, Eigen::nbThreads());

        std::string ext = config.input_file.extension();
        Wavefunction wfn;
        if ( ext == ".gjf" || ext == ".com") {
            wfn = run_from_gaussian_input_file(config);
        }
        else {
            wfn = run_from_xyz_file(config);
        }
        fs::path fchk_path = config.input_file;
        fchk_path.replace_extension(".fchk");
        occ::io::FchkWriter fchk(fchk_path.string());
        fchk.set_title(fmt::format("{} {}/{} generated by occ-ng", fchk_path.stem(), config.method, config.basis_name));
        fchk.set_method(config.method);
        fchk.set_basis_name(config.basis_name);
        wfn.save(fchk);
        fchk.write();

    } catch (const char *ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::string &ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::exception &ex) {
        fmt::print(error_format, ex.what());
        return 1;
    } catch (...) {
        fmt::print("Exception:\n- Unknown...\n");
        return 1;
    }
    occ::timing::stop(occ::timing::global);
    occ::timing::print_timings();
    return 0;
}
