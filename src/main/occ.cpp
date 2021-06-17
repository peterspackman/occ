#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/inputfile.h>
#include <occ/dft/dft.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/core/units.h>
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
using occ::dft::DFT;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

struct InputConfiguration {
    fs::path input_file;
    std::string method;
    std::string basis_name;
    std::optional<std::string> solvent{std::nullopt};
    size_t multiplicity;
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int charge;
    std::optional<std::string> solvent_surface_filename{std::nullopt};
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

This version of occ makes use of the following third party libraries:

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


template<typename T, SpinorbitalKind SK>
Wavefunction run_method(Molecule &m, const occ::qm::BasisSet &basis, const InputConfiguration &config)
{
    fmt::print("In run_method\n");
    if constexpr(std::is_same<T, DFT>::value)
    {
        fmt::print("Creating DFT\n");
        DFT ks(config.method, basis, m.atoms(), SK);
        fmt::print("Created DFT\n");
        SCF<DFT, SK> scf(ks);
        fmt::print("Created SCF\n");
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        scf.start_incremental_F_threshold = 0.0;
        double e = scf.compute_scf_energy();
        return scf.wavefunction();
    }
    else
    {
        fmt::print("Creating T\n");
        T proc(m.atoms(), basis);
        fmt::print("Created T\n");
        SCF<T, SK> scf(proc);
        fmt::print("Created SCF\n");
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        double e = scf.compute_scf_energy();
        return scf.wavefunction();

    }
}

template<typename T, SpinorbitalKind SK>
Wavefunction run_solvated_method(const Wavefunction &wfn, const InputConfiguration &config)
{
    using occ::solvent::SolvationCorrectedProcedure;
    if constexpr(std::is_same<T, DFT>::value)
    {
        DFT ks(config.method, wfn.basis, wfn.atoms, SK);
        SolvationCorrectedProcedure<DFT> proc_solv(ks, *config.solvent);
        SCF<SolvationCorrectedProcedure<DFT>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        scf.start_incremental_F_threshold = 0.0;
        scf.set_initial_guess_from_wfn(wfn);
        double e = scf.compute_scf_energy();
        if(config.solvent_surface_filename) proc_solv.write_surface_file(*config.solvent_surface_filename);
        return scf.wavefunction();
    }
    else
    {
        T proc(wfn.atoms, wfn.basis);
        SolvationCorrectedProcedure<T> proc_solv(proc, *config.solvent);
        SCF<SolvationCorrectedProcedure<T>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        scf.set_initial_guess_from_wfn(wfn);
        scf.start_incremental_F_threshold = 0.0;
        double e = scf.compute_scf_energy();
        if(config.solvent_surface_filename) proc_solv.write_surface_file(*config.solvent_surface_filename);
        return scf.wavefunction();

    }
}



void print_configuration(const Molecule &m, const InputConfiguration &config)
{
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", config.input_file.string(), "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    fmt::print("\n");

    fmt::print("Input method string: '{}'\n", config.method);
    fmt::print("Input basis name: '{}'\n", config.basis_name);
    fmt::print("Total charge: {}\n", config.charge);
    fmt::print("System multiplicity: {}\n", config.multiplicity);
}

occ::qm::BasisSet load_basis_set(const Molecule &m, const std::string &name)
{
    occ::qm::BasisSet basis(name, m.atoms());
    basis.set_pure(false);
    fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
    return basis;
}

Wavefunction run_from_xyz_file(const InputConfiguration &config)
{
    Molecule m = occ::chem::read_xyz_file(config.input_file.string());
    print_configuration(m, config);

    auto basis = load_basis_set(m, config.basis_name);

    if (config.method == "rhf")
        return run_method<HartreeFock, SpinorbitalKind::Restricted>(m, basis, config);
    else if (config.method == "ghf")
        return run_method<HartreeFock, SpinorbitalKind::General>(m, basis, config);
    else if (config.method == "uhf")
        return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(m, basis, config);
    else
    {
        if (config.spinorbital_kind == SpinorbitalKind::Unrestricted) {
            return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis, config);
        }
        else {
            return run_method<DFT, SpinorbitalKind::Restricted>(m, basis, config);
        }
    }

}

Wavefunction run_from_gaussian_input_file(InputConfiguration &config)
{
    occ::io::GaussianInputFile com(config.input_file.string());
    Molecule m(com.atomic_numbers, com.atomic_positions);

    config.method = com.method;
    config.charge = com.charge;
    config.multiplicity = com.multiplicity;
    config.basis_name = com.basis_name;

    print_configuration(m, config);
    auto basis = load_basis_set(m, com.basis_name);

    if(com.method_type == occ::io::GaussianInputFile::MethodType::HF)
    {
        if(com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
            return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(m, basis, config);
        else
            return run_method<HartreeFock, SpinorbitalKind::Restricted>(m, basis, config);
    }
    else {
        if(com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
            return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis, config);
        else
            return run_method<DFT, SpinorbitalKind::Restricted>(m, basis, config);
    }
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

    parser.add_argument("--solvent")
        .help("Solvent name");

    parser.add_argument("--solvent-file")
        .help("Solvent surface filename");

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

        if(auto solvent = parser.present("--solvent"))
        {
            double esolv{0.0};
            config.solvent = *solvent;
            config.solvent_surface_filename = parser.present("--solvent-file");
            Wavefunction wfn2;
            if(config.method == "ghf")
            {
                fmt::print("Hartree-Fock + SMD with general spinorbitals\n");
                wfn2 = run_solvated_method<HartreeFock, SpinorbitalKind::General>(wfn, config);
                esolv = wfn2.energy.total - wfn.energy.total;
            }
            else if(config.method == "rhf")
            {
                fmt::print("Hartree-Fock + SMD with restricted spinorbitals\n");
                wfn2 = run_solvated_method<HartreeFock, SpinorbitalKind::Restricted>(wfn, config);
                esolv = wfn2.energy.total - wfn.energy.total;
            }
            else if(config.method == "uhf")
            {
                fmt::print("Hartree-Fock + SMD with unrestricted spinorbitals\n");
                wfn2 = run_solvated_method<HartreeFock, SpinorbitalKind::Unrestricted>(wfn, config);
                esolv = wfn2.energy.total - wfn.energy.total;
            }
            else
            {
                if(config.spinorbital_kind == SpinorbitalKind::Restricted)
                {
                    fmt::print("Kohn-Sham DFT + SMD with restricted spinorbitals\n");
                    wfn2 = run_solvated_method<DFT, SpinorbitalKind::Restricted>(wfn, config);
                    esolv = wfn2.energy.total - wfn.energy.total;
                }
                else
                {
                    fmt::print("Kohn-Sham DFT + SMD with unrestricted spinorbitals\n");
                    wfn2 = run_solvated_method<DFT, SpinorbitalKind::Unrestricted>(wfn, config);
                    esolv = wfn2.energy.total - wfn.energy.total;
                }
            }

            fmt::print("Estimated delta G(solv) {:20.12f} ({:.3f} kJ/mol, {:.3f} kcal/mol)\n", 
                        esolv, esolv * occ::units::AU_TO_KJ_PER_MOL, esolv * occ::units::AU_TO_KCAL_PER_MOL);

            fchk_path.replace_extension(".solvated.fchk");
            occ::io::FchkWriter fchk_solv(fchk_path.string());
            fchk_solv.set_title(fmt::format("{} {}/{} generated by occ-ng, SMD solvent = {}",
                           fchk_path.stem(), config.method, config.basis_name, *config.solvent));
            fchk_solv.set_method(config.method);
            fchk_solv.set_basis_name(config.basis_name);
            wfn2.save(fchk_solv);
            fchk_solv.write();

        }


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
