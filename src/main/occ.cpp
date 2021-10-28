#include <boost/version.hpp>
#include <gemmi/version.hpp>
#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/constants.h>
#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/gaussian_input_file.h>
#include <occ/qm/hf.h>
#include <occ/qm/partitioning.h>
#include <occ/qm/scf.h>
#include <occ/solvent/solvation_correction.h>
#include <spdlog/cfg/env.h>
#include <xc.h>

namespace fs = std::filesystem;
using occ::chem::Element;
using occ::chem::Molecule;
using occ::dft::DFT;
using occ::hf::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;

struct InputConfiguration {
    fs::path input_file;
    std::string method;
    std::string basis_name;
    bool spherical{false};
    std::optional<std::string> solvent{std::nullopt};
    std::optional<std::string> density_fitting_basis_name{std::nullopt};
    size_t multiplicity;
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int charge;
    std::optional<std::string> solvent_surface_filename{std::nullopt};
};

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

)",
               eigen_version_string, libint_version_string, xc_version_string,
               gemmi_version_string, boost_version_string, fmt_version_string,
               spdlog_version_string);
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_method(Molecule &m, const occ::qm::BasisSet &basis,
                        const InputConfiguration &config) {

    T proc = [&]() {
        if constexpr (std::is_same<T, DFT>::value)
            return T(config.method, basis, m.atoms(), SK);
        else
            return T(m.atoms(), basis);
    }();

    if (config.density_fitting_basis_name)
        proc.set_density_fitting_basis(*config.density_fitting_basis_name);
    SCF<T, SK> scf(proc);
    scf.set_charge_multiplicity(config.charge, config.multiplicity);

    double e = scf.compute_scf_energy();
    Wavefunction wfn = scf.wavefunction();
    constexpr bool calc_properties{false};
    if(calc_properties) {
        fmt::print("\nproperties\n----------\n");
        occ::Vec3 origin = m.center_of_mass() * occ::units::ANGSTROM_TO_BOHR;
        fmt::print("center of mass (bohr)        {:12.6f} {:12.6f} {:12.6f}\n\n",
                   origin(0), origin(1), origin(2));
        auto e_mult =
            proc.template compute_electronic_multipoles<3>(SK, wfn.D, origin);
        auto nuc_mult = proc.template compute_nuclear_multipoles<3>(origin);
        auto tot_mult = e_mult + nuc_mult;
        fmt::print("electronic multipole\n{}\n", e_mult);
        fmt::print("nuclear multipole\n{}\n", nuc_mult);
        fmt::print("total multipole\n{}\n", tot_mult);
        occ::Vec mulliken_charges =
            -2 * occ::qm::mulliken_partition<SK>(proc.basis(), proc.atoms(), wfn.D,
                                                 proc.compute_overlap_matrix());
        fmt::print("Mulliken charges\n");
        for (size_t i = 0; i < wfn.atoms.size(); i++) {
            mulliken_charges(i) += wfn.atoms[i].atomic_number;
            fmt::print("Atom {}: {:12.6f}\n", i, mulliken_charges(i));
        }
    }
    return wfn;
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_solvated_method(const Wavefunction &wfn,
                                 const InputConfiguration &config) {
    using occ::solvent::SolvationCorrectedProcedure;
    if constexpr (std::is_same<T, DFT>::value) {
        DFT ks(config.method, wfn.basis, wfn.atoms, SK);
        SolvationCorrectedProcedure<DFT> proc_solv(ks, *config.solvent);
        SCF<SolvationCorrectedProcedure<DFT>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        scf.start_incremental_F_threshold = 0.0;
        scf.set_initial_guess_from_wfn(wfn);
        double e = scf.compute_scf_energy();
        if (config.solvent_surface_filename)
            proc_solv.write_surface_file(*config.solvent_surface_filename);
        return scf.wavefunction();
    } else {
        T proc(wfn.atoms, wfn.basis);
        SolvationCorrectedProcedure<T> proc_solv(proc, *config.solvent);
        SCF<SolvationCorrectedProcedure<T>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.charge, config.multiplicity);
        scf.set_initial_guess_from_wfn(wfn);
        scf.start_incremental_F_threshold = 0.0;
        double e = scf.compute_scf_energy();
        if (config.solvent_surface_filename)
            proc_solv.write_surface_file(*config.solvent_surface_filename);
        return scf.wavefunction();
    }
}

void print_configuration(const Molecule &m, const InputConfiguration &config) {
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n",
               config.input_file.string(), "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                   Element(atom.atomic_number).symbol(), atom.x, atom.y,
                   atom.z);
    }
    fmt::print("\n");

    fmt::print("Input method string: '{}'\n", config.method);
    fmt::print("Input basis name: '{}'\n", config.basis_name);
    fmt::print("Total charge: {}\n", config.charge);
    fmt::print("System multiplicity: {}\n", config.multiplicity);
    fmt::print("Inertia Tensor (x 10^-46 kg m^2)\n{}\n\n", m.inertia_tensor());
    fmt::print("Principal moments of inertia\n{}\n\n",
               m.principal_moments_of_inertia());
    fmt::print("Rotational constants (GHz)\n{}\n\n", m.rotational_constants());
    fmt::print("Rotational free energy      {: 12.6f} kJ/mol\n",
               m.rotational_free_energy(occ::constants::celsius<double> + 25));
    fmt::print(
        "Translational free energy   {: 12.6f} kJ/mol\n",
        m.translational_free_energy(occ::constants::celsius<double> + 25));
}

occ::qm::BasisSet load_basis_set(const Molecule &m, const std::string &name,
                                 bool spherical) {
    occ::qm::BasisSet basis(name, m.atoms());
    basis.set_pure(spherical);
    fmt::print("Loaded basis set: {}\n", spherical ? "spherical" : "cartesian");
    fmt::print("number of shells:            {}\n", basis.size());
    fmt::print("number of  basis functions:  {}\n", basis.nbf());
    fmt::print("max angular momentum:        {}\n", basis.max_l());
    return basis;
}

Wavefunction run_from_xyz_file(const InputConfiguration &config) {
    Molecule m = occ::chem::read_xyz_file(config.input_file.string());
    print_configuration(m, config);

    auto basis = load_basis_set(m, config.basis_name, config.spherical);

    if (config.method == "rhf")
        return run_method<HartreeFock, SpinorbitalKind::Restricted>(m, basis,
                                                                    config);
    else if (config.method == "ghf")
        return run_method<HartreeFock, SpinorbitalKind::General>(m, basis,
                                                                 config);
    else if (config.method == "uhf")
        return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(m, basis,
                                                                      config);
    else {
        if (config.spinorbital_kind == SpinorbitalKind::Unrestricted) {
            return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis,
                                                                  config);
        } else {
            return run_method<DFT, SpinorbitalKind::Restricted>(m, basis,
                                                                config);
        }
    }
}

Wavefunction run_from_gaussian_input_file(InputConfiguration &config) {
    occ::io::GaussianInputFile com(config.input_file.string());
    Molecule m(com.atomic_numbers, com.atomic_positions);

    config.method = com.method;
    config.charge = com.charge;
    config.multiplicity = com.multiplicity;
    config.basis_name = com.basis_name;

    print_configuration(m, config);
    auto basis = load_basis_set(m, com.basis_name, false);

    if (com.method_type == occ::io::GaussianInputFile::MethodType::HF) {
        if (com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
            return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(
                m, basis, config);
        else
            return run_method<HartreeFock, SpinorbitalKind::Restricted>(
                m, basis, config);
    } else {
        if (com.spinorbital_kind() == SpinorbitalKind::Unrestricted)
            return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis,
                                                                  config);
        else
            return run_method<DFT, SpinorbitalKind::Restricted>(m, basis,
                                                                config);
    }
}

int main(int argc, char *argv[]) {
    occ::timing::start(occ::timing::category::global);
    occ::timing::start(occ::timing::category::io);
    cxxopts::Options options("occ", "A program for quantum chemistry");
    options.positional_help("[input_file] [method] [basis]")
        .show_positional_help();

    options.add_options()("h,help", "Print help")(
        "i,input", "Input file", cxxopts::value<std::string>())(
        "b,basis", "Basis set name",
        cxxopts::value<std::string>()->default_value("3-21G"))(
        "d,df-basis", "Basis set name", cxxopts::value<std::string>())(
        "t,threads", "Number of threads",
        cxxopts::value<int>()->default_value("1"))(
        "m,method", "QM method",
        cxxopts::value<std::string>()->default_value("rhf"))(
        "c,charge", "System net charge",
        cxxopts::value<int>()->default_value("0"))(
        "n,multiplicity", "System multiplicity",
        cxxopts::value<int>()->default_value("1"))(
        "u,unrestricted", "Use unrestricted DFT",
        cxxopts::value<bool>()->default_value("false"))(
        "v,verbosity", "Logging verbosity",
        cxxopts::value<std::string>()->default_value("WARN"))(
        "spherical", "Use spherical basis functions",
        cxxopts::value<bool>()->default_value("false"))(
        "s,solvent", "Use SMD solvation model with solvent",
        cxxopts::value<std::string>())("f,solvent-file",
                                       "Write solvation surface to file",
                                       cxxopts::value<std::string>());

    options.parse_positional({"input", "method", "basis"});

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        fmt::print("{}\n", options.help());
        exit(0);
    }

    auto level = occ::log::level::warn;
    if (result.count("verbosity")) {
        std::string level_lower =
            occ::util::to_lower_copy(result["verbosity"].as<std::string>());
        if (level_lower == "debug")
            level = occ::log::level::debug;
        else if (level_lower == "info")
            level = occ::log::level::info;
        else if (level_lower == "error")
            level = occ::log::level::err;
    }
    occ::log::set_level(level);
    spdlog::set_level(level);
    print_header();
    occ::timing::stop(occ::timing::category::io);

    const std::string error_format =
        "Exception:\n    {}\nTerminating program.\n";

    if (result.count("input") == 0) {
        occ::log::error("must provide an input file!");
        exit(1);
    }

    try {
        libint2::Shell::do_enforce_unit_normalization(false);
        libint2::initialize();
        occ::timing::start(occ::timing::category::io);
        InputConfiguration config;
        config.input_file = result["input"].as<std::string>();
        config.basis_name = result["basis"].as<std::string>();
        config.multiplicity = result["multiplicity"].as<int>();
        config.method = result["method"].as<std::string>();
        config.charge = result["charge"].as<int>();
        config.spherical = result["spherical"].as<bool>();
        if (result.count("df-basis"))
            config.density_fitting_basis_name =
                result["df-basis"].as<std::string>();
        if (config.multiplicity != 1 || result.count("unrestricted") ||
            config.method == "uhf") {
            config.spinorbital_kind = SpinorbitalKind::Unrestricted;
            fmt::print("unrestricted spinorbital kind\n");
        } else if (config.method == "ghf") {
            config.spinorbital_kind = SpinorbitalKind::General;
            fmt::print("general spinorbital kind\n");
        } else {
            fmt::print("restricted spinorbital kind\n");
        }

        using occ::parallel::nthreads;
        nthreads = result["threads"].as<int>();
        fmt::print("\nParallelization: {} threads, {} Eigen threads\n",
                   nthreads, Eigen::nbThreads());

        std::string ext = config.input_file.extension();
        occ::timing::stop(occ::timing::category::io);
        Wavefunction wfn = [&ext, &config]() {
            if (ext == ".gjf" || ext == ".com") {
                return run_from_gaussian_input_file(config);
            } else {
                return run_from_xyz_file(config);
            }
        }();

        fs::path fchk_path = config.input_file;
        fchk_path.replace_extension(".fchk");
        fs::path npz_path = config.input_file;
        npz_path.replace_extension(".npz");
        occ::io::FchkWriter fchk(fchk_path.string());
        fchk.set_title(fmt::format("{} {}/{} generated by occ",
                                   fchk_path.stem(), config.method,
                                   config.basis_name));
        fchk.set_method(config.method);
        fchk.set_basis_name(config.basis_name);
        wfn.save(fchk);
        fchk.write();
        fmt::print("wavefunction stored in {}\n", fchk_path);
        if (config.spherical)
            occ::log::warn("Spherical basis coefficients and ordering are not "
                           "yet implemented for Fchk files");

        if (result.count("solvent")) {
            double esolv{0.0};
            config.solvent = result["solvent"].as<std::string>();
            if (result.count("solvent-file"))
                config.solvent_surface_filename =
                    result["solvent-file"].as<std::string>();
            Wavefunction wfn2 = [&config, &wfn]() {
                if (config.method == "ghf") {
                    fmt::print(
                        "Hartree-Fock + SMD with general spinorbitals\n");
                    return run_solvated_method<HartreeFock,
                                               SpinorbitalKind::General>(
                        wfn, config);
                } else if (config.method == "rhf") {
                    fmt::print(
                        "Hartree-Fock + SMD with restricted spinorbitals\n");
                    return run_solvated_method<HartreeFock,
                                               SpinorbitalKind::Restricted>(
                        wfn, config);
                } else if (config.method == "uhf") {
                    fmt::print(
                        "Hartree-Fock + SMD with unrestricted spinorbitals\n");
                    return run_solvated_method<HartreeFock,
                                               SpinorbitalKind::Unrestricted>(
                        wfn, config);
                } else {
                    if (config.spinorbital_kind ==
                        SpinorbitalKind::Restricted) {
                        fmt::print("Kohn-Sham DFT + SMD with restricted "
                                   "spinorbitals\n");
                        return run_solvated_method<DFT,
                                                   SpinorbitalKind::Restricted>(
                            wfn, config);
                    } else {
                        fmt::print("Kohn-Sham DFT + SMD with unrestricted "
                                   "spinorbitals\n");
                        return run_solvated_method<
                            DFT, SpinorbitalKind::Unrestricted>(wfn, config);
                    }
                }
            }();
            esolv = wfn2.energy.total - wfn.energy.total;

            fmt::print("Estimated delta G(solv) {:20.12f} ({:.3f} kJ/mol, "
                       "{:.3f} kcal/mol)\n",
                       esolv, esolv * occ::units::AU_TO_KJ_PER_MOL,
                       esolv * occ::units::AU_TO_KCAL_PER_MOL);

            fchk_path.replace_extension(".solvated.fchk");
            occ::io::FchkWriter fchk_solv(fchk_path.string());
            fchk_solv.set_title(fmt::format(
                "{} {}/{} generated by occ, SMD solvent = {}", fchk_path.stem(),
                config.method, config.basis_name, *config.solvent));
            fchk_solv.set_method(config.method);
            fchk_solv.set_basis_name(config.basis_name);
            wfn2.save(fchk_solv);
            fchk_solv.write();
            fmt::print("solvated wavefunction stored in {}\n", fchk_path);
            if (config.spherical)
                occ::log::warn("Spherical basis coefficients and ordering are "
                               "not yet implemented for Fchk files");
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
