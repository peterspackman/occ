#include <occ/core/constants.h>
#include <occ/dft/dft.h>
#include <occ/io/occ_input.h>
#include <occ/main/single_point.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>

namespace occ::main {

using occ::core::Element;
using occ::core::Molecule;
using occ::dft::DFT;
using occ::io::OccInput;
using occ::qm::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;

void print_matrix_xyz(const Mat &m) {
    for (size_t i = 0; i < 3; i++) {
        log::info("{: 12.6f} {: 12.6f} {: 12.6f}", m(i, 0), m(i, 1), m(i, 2));
    }
}

void print_vector(const Vec3 &m) {
    log::info("{: 12.6f} {: 12.6f} {: 12.6f}", m(0), m(1), m(2));
}

occ::qm::AOBasis load_basis_set(const Molecule &m, const std::string &name,
                                bool spherical) {
    auto basis = occ::qm::AOBasis::load(m.atoms(), name);
    basis.set_pure(spherical);
    log::info("Loaded basis set: {}", spherical ? "spherical" : "cartesian");
    log::info("Number of shells:            {}", basis.size());
    log::info("Number of  basis functions:  {}", basis.nbf());
    log::info("Maximum angular momentum:    {}", basis.l_max());
    return basis;
}

void print_configuration(const Molecule &m, const OccInput &config) {
    log::info("{:=^72s}", "  Input  ");

    log::info("{: <20s} {: >20s}", "Method string", config.method.name);
    log::info("{: <20s} {: >20s}", "Basis name", config.basis.name);
    log::info("{: <20s} {: >20s}", "Shell kind",
              config.basis.spherical ? "spherical" : "Cartesian");
    log::info("{: <20s} {: >20d}", "Net charge",
              static_cast<int>(config.electronic.charge));
    log::info("{: <20s} {: >20d}", "Multiplicity",
              config.electronic.multiplicity);

    log::info("{:-<72s}", fmt::format("Geometry '{}' (au)  ", config.filename));
    for (const auto &atom : m.atoms()) {
        log::info("{:^3s} {:12.6f} {:12.6f} {:12.6f}",
                  Element(atom.atomic_number).symbol(), atom.x, atom.y, atom.z);
    }

    double temperature = occ::constants::celsius<double> + 25;

    log::info("{:-<72s}", "Inertia tensor (x 10e-46 kg m^2)  ");
    print_matrix_xyz(m.inertia_tensor());
    log::info("{:-<72s}", "Principal moments of inertia  ");
    print_vector(m.principal_moments_of_inertia());
    log::info("{:-<72s}", "Rotational constants (GHz)  ");
    print_vector(m.rotational_constants());
    log::info("\n");

    log::info("{:-<72s}",
              fmt::format("Gas-phase properties (at {} K)  ", temperature));
    log::info("Rotational free energy      {: 12.6f} kJ/mol",
              m.rotational_free_energy(temperature));
    log::info("Translational free energy   {: 12.6f} kJ/mol",
              m.translational_free_energy(temperature));
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_method(Molecule &m, const occ::qm::AOBasis &basis,
                        const OccInput &config) {

    T proc = [&]() {
        if constexpr (std::is_same<T, DFT>::value)
            return T(config.method.name, basis, config.method.dft_grid);
        else
            return T(basis);
    }();

    if (!config.basis.df_name.empty())
        proc.set_density_fitting_basis(config.basis.df_name);
    SCF<T> scf(proc, SK);
    occ::log::trace("Setting system charge: {}", config.electronic.charge);
    occ::log::trace("Setting system multiplicity: {}",
                    config.electronic.multiplicity);
    scf.set_charge_multiplicity(config.electronic.charge,
                                config.electronic.multiplicity);
    if (!config.basis.df_name.empty())
        scf.start_incremental_F_threshold = 0.0;

    double e = scf.compute_scf_energy();
    if constexpr (std::is_same<T, DFT>::value) {
        double enlc = proc.post_scf_nlc_correction(scf.mo);
        if (enlc != 0.0) {
            log::info("Post SCF NLC correction:         {: 20.12f}", enlc);
            e += enlc;
            log::info("Corrected total energy:          {: 20.12f}", e);
        }
    }
    return scf.wavefunction();
}

template <typename T, SpinorbitalKind SK>
Wavefunction run_solvated_method(const Wavefunction &wfn,
                                 const OccInput &config) {
    using occ::solvent::SolvationCorrectedProcedure;
    if constexpr (std::is_same<T, DFT>::value) {
        DFT ks(config.method.name, wfn.basis, config.method.dft_grid);
        if (!config.basis.df_name.empty())
            ks.set_density_fitting_basis(config.basis.df_name);
        SolvationCorrectedProcedure<DFT> proc_solv(ks,
                                                   config.solvent.solvent_name);
        SCF<SolvationCorrectedProcedure<DFT>> scf(proc_solv, SK);
        scf.set_charge_multiplicity(config.electronic.charge,
                                    config.electronic.multiplicity);
        scf.start_incremental_F_threshold = 0.0;
        scf.set_initial_guess_from_wfn(wfn);
        double e = scf.compute_scf_energy();
        if (!config.solvent.output_surface_filename.empty())
            proc_solv.write_surface_file(
                config.solvent.output_surface_filename);
        return scf.wavefunction();
    } else {
        T proc(wfn.basis);
        if (!config.basis.df_name.empty())
            proc.set_density_fitting_basis(config.basis.df_name);
        SolvationCorrectedProcedure<T> proc_solv(proc,
                                                 config.solvent.solvent_name);
        SCF<SolvationCorrectedProcedure<T>> scf(proc_solv, SK);
        scf.set_charge_multiplicity(config.electronic.charge,
                                    config.electronic.multiplicity);
        scf.set_initial_guess_from_wfn(wfn);
        scf.start_incremental_F_threshold = 0.0;
        double e = scf.compute_scf_energy();
        if (!config.solvent.output_surface_filename.empty())
            proc_solv.write_surface_file(
                config.solvent.output_surface_filename);
        return scf.wavefunction();
    }
}

Wavefunction
single_point_driver(const OccInput &config,
                    const std::optional<Wavefunction> &guess = {}) {
    Molecule m = config.geometry.molecule();
    print_configuration(m, config);

    auto basis = load_basis_set(m, config.basis.name, config.basis.spherical);

    if (config.solvent.solvent_name.empty()) {
        if (config.method.name == "rhf")
            return run_method<HartreeFock, SpinorbitalKind::Restricted>(
                m, basis, config);
        else if (config.method.name == "ghf")
            return run_method<HartreeFock, SpinorbitalKind::General>(m, basis,
                                                                     config);
        else if (config.method.name == "uhf")
            return run_method<HartreeFock, SpinorbitalKind::Unrestricted>(
                m, basis, config);
        else {
            if (config.electronic.spinorbital_kind ==
                SpinorbitalKind::Unrestricted) {
                return run_method<DFT, SpinorbitalKind::Unrestricted>(m, basis,
                                                                      config);
            } else {
                return run_method<DFT, SpinorbitalKind::Restricted>(m, basis,
                                                                    config);
            }
        }
    } else {
        if (config.method.name == "ghf") {
            log::info("Hartree-Fock + SMD with general spinorbitals");
            return run_solvated_method<HartreeFock, SpinorbitalKind::General>(
                *guess, config);
        } else if (config.method.name == "rhf") {
            log::info("Hartree-Fock + SMD with restricted spinorbitals");
            return run_solvated_method<HartreeFock,
                                       SpinorbitalKind::Restricted>(*guess,
                                                                    config);
        } else if (config.method.name == "uhf") {
            log::info("Hartree-Fock + SMD with unrestricted spinorbitals");
            return run_solvated_method<HartreeFock,
                                       SpinorbitalKind::Unrestricted>(*guess,
                                                                      config);
        } else {
            if (config.electronic.spinorbital_kind ==
                SpinorbitalKind::Restricted) {
                log::info("Kohn-Sham DFT + SMD with restricted "
                          "spinorbitals");
                return run_solvated_method<DFT, SpinorbitalKind::Restricted>(
                    *guess, config);
            } else {
                log::info("Kohn-Sham DFT + SMD with unrestricted "
                          "spinorbitals");
                return run_solvated_method<DFT, SpinorbitalKind::Unrestricted>(
                    *guess, config);
            }
        }
    }
}

Wavefunction single_point_calculation(const OccInput &config) {
    return single_point_driver(config);
}

Wavefunction single_point_calculation(const OccInput &config,
                                      const Wavefunction &wfn) {
    return single_point_driver(config, wfn);
}

} // namespace occ::main
