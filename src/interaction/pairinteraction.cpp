#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/qm/hf.h>
#include <occ/qm/merge.h>
#include <occ/xdm/xdm.h>

namespace occ::interaction {
using qm::HartreeFock;
using qm::SpinorbitalKind;

CEModelInteraction::CEModelInteraction(const CEParameterizedModel &facs)
    : scale_factors(facs) {}

template <SpinorbitalKind kind, typename Proc = qm::HartreeFock>
void compute_ce_model_energies(Wavefunction &wfn, Proc &proc, double precision,
                               const Mat &Schwarz) {
    if (wfn.have_energies) {
        occ::log::debug("Already have monomer energies, skipping");
        return;
    }
    using occ::qm::expectation;
    using occ::qm::matrix_dimensions;
    if constexpr (kind == SpinorbitalKind::Restricted) {
        wfn.V = proc.compute_nuclear_attraction_matrix();
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.mo.D, wfn.V);
        wfn.T = proc.compute_kinetic_matrix();
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.mo.D, wfn.T);
        wfn.H = wfn.V + wfn.T;
        std::tie(wfn.J, wfn.K) = proc.compute_JK(wfn.mo, precision, Schwarz);
        wfn.energy.coulomb = expectation<kind>(wfn.mo.D, wfn.J);
        if constexpr (std::is_same<Proc, dft::DFT>::value) {
            wfn.energy.exchange = proc.exchange_energy_total();
        } else {
            wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
        }

        wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
        wfn.energy.nuclear_repulsion = proc.nuclear_repulsion_energy();
        if (proc.have_effective_core_potentials()) {
            wfn.Vecp = proc.compute_effective_core_potential_matrix();
            wfn.H += wfn.Vecp;
            wfn.energy.ecp = 2 * expectation<kind>(wfn.mo.D, wfn.Vecp);
        }
        wfn.energy.core = 2 * expectation<kind>(wfn.mo.D, wfn.H);
    } else {
        namespace block = occ::qm::block;
        size_t rows, cols;
        std::tie(rows, cols) =
            matrix_dimensions<SpinorbitalKind::Unrestricted>(wfn.nbf);
        wfn.T = Mat(rows, cols);
        wfn.V = Mat(rows, cols);
        block::a(wfn.T) = proc.compute_kinetic_matrix();
        block::b(wfn.T) = block::a(wfn.T);
        block::a(wfn.V) = proc.compute_nuclear_attraction_matrix();
        block::b(wfn.V) = block::a(wfn.V);
        wfn.H = wfn.V + wfn.T;
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.mo.D, wfn.V);
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.mo.D, wfn.T);
        std::tie(wfn.J, wfn.K) = proc.compute_JK(wfn.mo, precision, Schwarz);
        wfn.energy.coulomb = expectation<kind>(wfn.mo.D, wfn.J);
        if constexpr (std::is_same<Proc, dft::DFT>::value) {
            wfn.energy.exchange = proc.exchange_energy_total();
        } else {
            wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
        }

        wfn.energy.nuclear_repulsion = proc.nuclear_repulsion_energy();

        if (proc.have_effective_core_potentials()) {
            wfn.Vecp = Mat(rows, cols);
            block::a(wfn.Vecp) = proc.compute_effective_core_potential_matrix();
            block::b(wfn.Vecp) = block::a(wfn.Vecp);
            wfn.H += wfn.Vecp;
            wfn.energy.ecp = 2 * expectation<kind>(wfn.mo.D, wfn.Vecp);
        }
        wfn.energy.core = 2 * expectation<kind>(wfn.mo.D, wfn.H);
    }
    wfn.have_energies = true;
}

void compute_xdm_parameters(Wavefunction &wfn) {
    if (wfn.have_xdm_parameters) {
        occ::log::debug("Skipping computation of parameters");
        return;
    }
    occ::log::debug("Computing xdm_parameters");
    occ::xdm::XDM xdm_calc(wfn.basis, wfn.charge());
    auto energy = xdm_calc.energy(wfn.mo);
    wfn.xdm_polarizabilities = xdm_calc.polarizabilities();
    wfn.xdm_moments = xdm_calc.moments();
    wfn.xdm_volumes = xdm_calc.atom_volume();
    wfn.xdm_free_volumes = xdm_calc.free_atom_volume();
    wfn.have_xdm_parameters = true;
    occ::log::debug("Computed xdm_parameters");
}

template <typename Proc>
void compute_ce_model_energies(Wavefunction &wfn, Proc &hf, double precision,
                               const Mat &Schwarz, bool xdm) {
    if (wfn.is_restricted()) {
        occ::log::debug("Restricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Restricted, Proc>(
            wfn, hf, precision, Schwarz);
    } else {
        occ::log::debug("Unrestricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Unrestricted, Proc>(
            wfn, hf, precision, Schwarz);
    }

    if (xdm) {
        compute_xdm_parameters(wfn);
    }
}

void compute_ce_model_energies(Wavefunction &wfn, HartreeFock &hf,
                               double precision, const Mat &Schwarz, bool xdm) {
    if (wfn.is_restricted()) {
        occ::log::debug("Restricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Restricted, HartreeFock>(
            wfn, hf, precision, Schwarz);
    } else {
        occ::log::debug("Unrestricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Unrestricted, HartreeFock>(
            wfn, hf, precision, Schwarz);
    }

    if (xdm) {
        compute_xdm_parameters(wfn);
    }
}

void CEModelInteraction::set_use_density_fitting(bool value) {
    m_use_density_fitting = value;
}

void dump_matrix(const Mat &matrix) {
    size_t maxdim = std::max(matrix.rows(), matrix.cols());
    Eigen::Index fields = 5;

    if (fields == 0)
        fields = matrix.rows();
    size_t n_block = static_cast<size_t>((matrix.cols() - 0.1) / fields) + 1;
    for (size_t block = 0; block < n_block; block++) {
        Eigen::Index f = fields * block;
        Eigen::Index l = std::min(f + fields, matrix.cols());
        fmt::print("{:8s}", " ");
        for (size_t j = f; j < l; j++) {
            fmt::print(" {:8d}", j);
        }
        fmt::print("\n");
        for (size_t i = 0; i < matrix.rows(); i++) {
            fmt::print("{:8d}", i);
            for (size_t j = f; j < l; j++) {
                fmt::print(" {:8.5f}", matrix(i, j));
            }
            fmt::print("\n");
        }
    }
}

void CEModelInteraction::compute_monomer_energies(Wavefunction &wfn) const {
    constexpr double precision = std::numeric_limits<double>::epsilon();
    HartreeFock hf(wfn.basis);
    if (m_use_density_fitting) {
        hf.set_density_fitting_basis("def2-universal-jkfit");
    }
    Mat schwarz = hf.compute_schwarz_ints();
    fmt::print("Calculating xdm parameters: {}\n", scale_factors.xdm);
    compute_ce_model_energies(wfn, hf, precision, schwarz, scale_factors.xdm);
}

CEEnergyComponents CEModelInteraction::operator()(Wavefunction &A,
                                                  Wavefunction &B) const {
    using occ::disp::ce_model_dispersion_energy;
    using occ::qm::Energy;
    constexpr double precision = std::numeric_limits<double>::epsilon();

    HartreeFock hf_a(A.basis);
    HartreeFock hf_b(B.basis);
    if (m_use_density_fitting) {
        hf_a.set_density_fitting_basis("def2-universal-jkfit");
        hf_b.set_density_fitting_basis("def2-universal-jkfit");
    }

    Mat schwarz_a = hf_a.compute_schwarz_ints();
    Mat schwarz_b = hf_b.compute_schwarz_ints();

    compute_ce_model_energies(A, hf_a, precision, schwarz_a, scale_factors.xdm);
    compute_ce_model_energies(B, hf_b, precision, schwarz_b, scale_factors.xdm);

    Wavefunction ABn(A, B);

    occ::log::debug("Merged wavefunction atoms:");
    for (const auto &a : ABn.atoms) {
        occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}", a.atomic_number,
                        a.x / occ::units::ANGSTROM_TO_BOHR,
                        a.y / occ::units::ANGSTROM_TO_BOHR,
                        a.z / occ::units::ANGSTROM_TO_BOHR);
    }

    // Can reuse the same HartreeFock object for both merged wfns: same basis
    // and atoms
    auto hf_AB = HartreeFock(ABn.basis);
    Mat schwarz_ab = hf_AB.compute_schwarz_ints();
    if (m_use_density_fitting) {
        hf_AB.set_density_fitting_basis("def2-universal-jkfit");
    }

    Wavefunction ABo = ABn;
    Mat S_AB = hf_AB.compute_overlap_matrix();
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();

    // no need to XDM for the combined wavefunctions
    compute_ce_model_energies(ABn, hf_AB, precision, schwarz_ab, false);
    compute_ce_model_energies(ABo, hf_AB, precision, schwarz_ab, false);

    occ::log::debug("ABn\n{}\n", ABn.energy);
    occ::log::debug("ABo\n{}\n", ABo.energy);

    Energy E_ABn = ABn.energy - (A.energy + B.energy);

    CEEnergyComponents energy;
    energy.coulomb = E_ABn.coulomb + E_ABn.nuclear_attraction +
                     E_ABn.nuclear_repulsion + E_ABn.ecp;
    occ::log::debug("Coulomb components:");
    occ::log::debug("ABn coulomb term {:20.12f}", E_ABn.coulomb);
    occ::log::debug("ABn en term      {:20.12f}", E_ABn.nuclear_attraction);
    occ::log::debug("ABn nn term      {:20.12f}", E_ABn.nuclear_repulsion);
    occ::log::debug("Total term       {:20.12f}", energy.coulomb);
    double eABn = ABn.energy.core + ABn.energy.exchange + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.exchange + ABo.energy.coulomb;
    double E_rep = eABo - eABn;
    energy.exchange_repulsion = E_ABn.exchange + E_rep;
    occ::log::debug("Exchange repulsion components:");
    occ::log::debug("ABn core term      {:20.12f}", ABn.energy.core);
    occ::log::debug("ABn exchange term  {:20.12f}", ABn.energy.exchange);
    occ::log::debug("ABn coulomb term   {:20.12f}", ABn.energy.coulomb);
    occ::log::debug("ABo core term      {:20.12f}", ABo.energy.core);
    occ::log::debug("ABo exchange term  {:20.12f}", ABo.energy.exchange);
    occ::log::debug("ABo coulomb term   {:20.12f}", ABo.energy.coulomb);
    occ::log::debug("E_rep term         {:20.12f}", E_rep);
    occ::log::debug("Exchange term      {:20.12f}", E_ABn.exchange);
    occ::log::debug("Total term         {:20.12f}", energy.exchange_repulsion);

    if (scale_factors.xdm) {
        fmt::print("XDM params: {} {}\n", scale_factors.xdm_a1,
                   scale_factors.xdm_a2);
        auto xdm_result = xdm::xdm_dispersion_interaction_energy(
            {A.atoms, A.xdm_polarizabilities, A.xdm_moments, A.xdm_volumes,
             A.xdm_free_volumes},
            {B.atoms, B.xdm_polarizabilities, B.xdm_moments, B.xdm_volumes,
             B.xdm_free_volumes},
            {scale_factors.xdm_a1, scale_factors.xdm_a2});
        energy.dispersion = std::get<0>(xdm_result);
    } else {
        energy.dispersion = ce_model_dispersion_energy(A.atoms, B.atoms);
    }
    energy.polarization = compute_polarization_energy(A, hf_a, B, hf_b);

    energy.total =
        scale_factors.scaled_total(energy.coulomb, energy.exchange_repulsion,
                                   energy.polarization, energy.dispersion);
    return energy;
}

CEEnergyComponents CEModelInteraction::dft_pair(const std::string &functional,
                                                Wavefunction &A,
                                                Wavefunction &B) const {
    using occ::dft::DFT;
    using occ::disp::ce_model_dispersion_energy;
    using occ::qm::Energy;
    constexpr double precision = std::numeric_limits<double>::epsilon();

    occ::dft::AtomGridSettings grid_settings{194, 50, 50, 1e-8};

    DFT dft_a(functional, A.basis);
    dft_a.set_integration_grid(grid_settings);
    DFT dft_b(functional, B.basis);
    dft_b.set_integration_grid(grid_settings);
    if (m_use_density_fitting) {
        dft_a.set_density_fitting_basis("def2-universal-jkfit");
        dft_b.set_density_fitting_basis("def2-universal-jkfit");
    }

    Mat schwarz_a = dft_a.compute_schwarz_ints();
    Mat schwarz_b = dft_b.compute_schwarz_ints();

    compute_ce_model_energies(A, dft_a, precision, schwarz_a,
                              scale_factors.xdm);
    compute_ce_model_energies(B, dft_b, precision, schwarz_b,
                              scale_factors.xdm);

    Wavefunction ABn(A, B);

    occ::log::debug("Merged wavefunction atoms:");
    for (const auto &a : ABn.atoms) {
        occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}", a.atomic_number,
                        a.x / occ::units::ANGSTROM_TO_BOHR,
                        a.y / occ::units::ANGSTROM_TO_BOHR,
                        a.z / occ::units::ANGSTROM_TO_BOHR);
    }

    // Can reuse the same HartreeFock object for both merged wfns: same basis
    // and atoms
    auto dft_AB = DFT(functional, ABn.basis);
    dft_AB.set_integration_grid(grid_settings);
    Mat schwarz_ab = dft_AB.compute_schwarz_ints();
    if (m_use_density_fitting) {
        dft_AB.set_density_fitting_basis("def2-universal-jkfit");
    }

    Wavefunction ABo = ABn;
    Mat S_AB = dft_AB.compute_overlap_matrix();
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();

    // no need to XDM for the combined wavefunctions
    compute_ce_model_energies(ABn, dft_AB, precision, schwarz_ab, false);
    compute_ce_model_energies(ABo, dft_AB, precision, schwarz_ab, false);

    occ::log::debug("ABn\n{}\n", ABn.energy);
    occ::log::debug("ABo\n{}\n", ABo.energy);

    Energy E_ABn = ABn.energy - (A.energy + B.energy);

    CEEnergyComponents energy;
    energy.coulomb = E_ABn.coulomb + E_ABn.nuclear_attraction +
                     E_ABn.nuclear_repulsion + E_ABn.ecp;
    occ::log::debug("Coulomb components:");
    occ::log::debug("ABn coulomb term {:20.12f}", E_ABn.coulomb);
    occ::log::debug("ABn en term      {:20.12f}", E_ABn.nuclear_attraction);
    occ::log::debug("ABn nn term      {:20.12f}", E_ABn.nuclear_repulsion);
    occ::log::debug("Total term       {:20.12f}", energy.coulomb);
    double eABn = ABn.energy.core + ABn.energy.exchange + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.exchange + ABo.energy.coulomb;
    double E_rep = eABo - eABn;
    energy.exchange_repulsion = E_ABn.exchange + E_rep;
    occ::log::debug("Exchange repulsion components:");
    occ::log::debug("ABn core term      {:20.12f}", ABn.energy.core);
    occ::log::debug("ABn exchange term  {:20.12f}", ABn.energy.exchange);
    occ::log::debug("ABn coulomb term   {:20.12f}", ABn.energy.coulomb);
    occ::log::debug("ABo core term      {:20.12f}", ABo.energy.core);
    occ::log::debug("ABo exchange term  {:20.12f}", ABo.energy.exchange);
    occ::log::debug("ABo coulomb term   {:20.12f}", ABo.energy.coulomb);
    occ::log::debug("E_rep term         {:20.12f}", E_rep);
    occ::log::debug("Exchange term      {:20.12f}", E_ABn.exchange);
    occ::log::debug("Total term         {:20.12f}", energy.exchange_repulsion);

    if (scale_factors.xdm) {
        fmt::print("XDM params: {} {}\n", scale_factors.xdm_a1,
                   scale_factors.xdm_a2);
        auto xdm_result = xdm::xdm_dispersion_interaction_energy(
            {A.atoms, A.xdm_polarizabilities, A.xdm_moments, A.xdm_volumes,
             A.xdm_free_volumes},
            {B.atoms, B.xdm_polarizabilities, B.xdm_moments, B.xdm_volumes,
             B.xdm_free_volumes},
            {scale_factors.xdm_a1, scale_factors.xdm_a2});
        energy.dispersion = std::get<0>(xdm_result);
    } else {
        energy.dispersion = ce_model_dispersion_energy(A.atoms, B.atoms);
    }
    energy.polarization = compute_polarization_energy(A, dft_a, B, dft_b);

    energy.total =
        scale_factors.scaled_total(energy.coulomb, energy.exchange_repulsion,
                                   energy.polarization, energy.dispersion);
    return energy;
}

} // namespace occ::interaction
