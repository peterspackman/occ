#include <occ/main/single_point.h>
#include <occ/qm/wavefunction.h>
#include <occ/io/occ_input.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/dft/dft.h>
#include <occ/qm/scf.h>
#include <occ/core/constants.h>

namespace occ::main {

using occ::core::Element;
using occ::core::Molecule;
using occ::dft::DFT;
using occ::hf::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::io::OccInput;

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

void print_configuration(const Molecule &m, const OccInput &config) {
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n",
               config.filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                   Element(atom.atomic_number).symbol(), atom.x, atom.y,
                   atom.z);
    }
    fmt::print("\n");

    fmt::print("Input method string: '{}'\n", config.method.name);
    fmt::print("Input basis name: '{}'\n", config.basis.name);
    fmt::print("Total charge: {}\n", config.electronic.charge);
    fmt::print("System multiplicity: {}\n", config.electronic.multiplicity);
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


template <typename T, SpinorbitalKind SK>
Wavefunction run_method(Molecule &m, const occ::qm::BasisSet &basis,
                        const OccInput &config) {

    T proc = [&]() {
        if constexpr (std::is_same<T, DFT>::value)
            return T(config.method.name, basis, m.atoms(), SK);
        else
            return T(m.atoms(), basis);
    }();

    if (!config.basis.df_name.empty())
        proc.set_density_fitting_basis(config.basis.df_name);
    SCF<T, SK> scf(proc);
    scf.set_charge_multiplicity(config.electronic.charge, config.electronic.multiplicity);

    double e = scf.compute_scf_energy();
    return scf.wavefunction();
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_solvated_method(const Wavefunction &wfn,
                                 const OccInput &config) {
    using occ::solvent::SolvationCorrectedProcedure;
    if constexpr (std::is_same<T, DFT>::value) {
        DFT ks(config.method.name, wfn.basis, wfn.atoms, SK);
        SolvationCorrectedProcedure<DFT> proc_solv(ks, config.solvent.solvent_name);
        SCF<SolvationCorrectedProcedure<DFT>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.electronic.charge, config.electronic.multiplicity);
        scf.start_incremental_F_threshold = 0.0;
        scf.set_initial_guess_from_wfn(wfn);
        double e = scf.compute_scf_energy();
        if (!config.solvent.output_surface_filename.empty())
            proc_solv.write_surface_file(config.solvent.output_surface_filename);
        return scf.wavefunction();
    } else {
        T proc(wfn.atoms, wfn.basis);
        SolvationCorrectedProcedure<T> proc_solv(proc, config.solvent.solvent_name);
        SCF<SolvationCorrectedProcedure<T>, SK> scf(proc_solv);
        scf.set_charge_multiplicity(config.electronic.charge, config.electronic.multiplicity);
        scf.set_initial_guess_from_wfn(wfn);
        scf.start_incremental_F_threshold = 0.0;
        double e = scf.compute_scf_energy();
        if (!config.solvent.output_surface_filename.empty())
            proc_solv.write_surface_file(config.solvent.output_surface_filename);
        return scf.wavefunction();
    }
}


Wavefunction single_point_driver(const OccInput &config, const std::optional<Wavefunction>& guess = {}) {
    Molecule m = config.geometry.molecule();
    print_configuration(m, config);

    auto basis = load_basis_set(m, config.basis.name, config.basis.spherical);

    if(config.solvent.solvent_name.empty()) {
        if (config.method.name == "rhf")
            return run_method<HartreeFock, SpinorbitalKind::Restricted>(m, basis,
                                                                        config);
        else if (config.method.name == "ghf")
            return run_method<HartreeFock, SpinorbitalKind::General>(m, basis,
                                                                     config);
        else if (config.method.name == "uhf")
            return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(m, basis,
                                                                          config);
        else {
            if (config.electronic.spinorbital_kind == SpinorbitalKind::Unrestricted) {
                return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis,
                                                                      config);
            } else {
                return run_method<DFT, SpinorbitalKind::Restricted>(m, basis,
                                                                    config);
            }
        }
    }
    else {
        if (config.method.name == "ghf") {
            fmt::print(
                "Hartree-Fock + SMD with general spinorbitals\n");
            return run_solvated_method<HartreeFock,
                                       SpinorbitalKind::General>(
                *guess, config);
        } else if (config.method.name == "rhf") {
            fmt::print(
                "Hartree-Fock + SMD with restricted spinorbitals\n");
            return run_solvated_method<HartreeFock,
                                       SpinorbitalKind::Restricted>(
                *guess, config);
        } else if (config.method.name == "uhf") {
            fmt::print(
                "Hartree-Fock + SMD with unrestricted spinorbitals\n");
            return run_solvated_method<HartreeFock,
                                       SpinorbitalKind::Unrestricted>(
                *guess, config);
        } else {
            if (config.electronic.spinorbital_kind ==
                SpinorbitalKind::Restricted) {
                fmt::print("Kohn-Sham DFT + SMD with restricted "
                           "spinorbitals\n");
                return run_solvated_method<DFT,
                                           SpinorbitalKind::Restricted>(
                    *guess, config);
            } else {
                fmt::print("Kohn-Sham DFT + SMD with unrestricted "
                           "spinorbitals\n");
                return run_solvated_method<
                    DFT, SpinorbitalKind::Unrestricted>(*guess, config);
            }
        }
    }
}

Wavefunction single_point_calculation(const OccInput &config) {
    return single_point_driver(config);
}

Wavefunction single_point_calculation(const OccInput &config, const Wavefunction &wfn) {
    return single_point_driver(config, wfn);
}

}
