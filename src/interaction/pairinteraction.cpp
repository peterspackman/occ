#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/qm/hf.h>
#include <occ/qm/merge.h>
#include <occ/qm/orb.h>
#include <occ/xdm/xdm.h>

namespace occ::interaction {

using qm::HartreeFock;
using qm::SpinorbitalKind;

CEModelInteraction::CEModelInteraction(const CEParameterizedModel &facs)
    : m_scale_factors(facs) {}

namespace impl {
template <bool flag = false> void static_invalid_kind() {
    static_assert(flag, "Invalid spinorbital kind");
}
}

template<SpinorbitalKind kind>
void compute_ce_core_matrices(Wavefunction &wfn, HartreeFock &proc) {
  if constexpr (kind == SpinorbitalKind::Restricted) {
    wfn.V = proc.compute_nuclear_attraction_matrix();
    wfn.T = proc.compute_kinetic_matrix();
    wfn.H = wfn.V + wfn.T;
    if (proc.have_effective_core_potentials()) {
        wfn.Vecp = proc.compute_effective_core_potential_matrix();
        wfn.H += wfn.Vecp;
    }
  }
  else if constexpr(kind == SpinorbitalKind::Unrestricted) {
    namespace block = occ::qm::block;
    size_t rows, cols;
    std::tie(rows, cols) =
      occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(wfn.nbf);
    wfn.T = Mat(rows, cols);
    wfn.V = Mat(rows, cols);
    block::a(wfn.T) = proc.compute_kinetic_matrix();
    block::b(wfn.T) = block::a(wfn.T);
    block::a(wfn.V) = proc.compute_nuclear_attraction_matrix();
    block::b(wfn.V) = block::a(wfn.V);
    wfn.H = wfn.V + wfn.T;
    if (proc.have_effective_core_potentials()) {
      wfn.Vecp = Mat(rows, cols);
      block::a(wfn.Vecp) = proc.compute_effective_core_potential_matrix();
      block::b(wfn.Vecp) = block::a(wfn.Vecp);
      wfn.H += wfn.Vecp;
    }
  }
  else impl::static_invalid_kind();
}

template<SpinorbitalKind kind>
void compute_ce_core_energies(Wavefunction &wfn, HartreeFock &proc) {
    using occ::qm::expectation;
    wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.mo.D, wfn.V);
    wfn.energy.kinetic = 2 * expectation<kind>(wfn.mo.D, wfn.T);
    if (proc.have_effective_core_potentials()) {
        wfn.energy.ecp = 2 * expectation<kind>(wfn.mo.D, wfn.Vecp);
    }
    wfn.energy.core = 2 * expectation<kind>(wfn.mo.D, wfn.H);
    wfn.energy.nuclear_repulsion = proc.nuclear_repulsion_energy();
}

template<SpinorbitalKind kind>
void compute_ce_model_energies_int(Wavefunction &wfn1, Wavefunction &wfn2,
                                   HartreeFock &proc,
                                   const CEMonomerCalculationParameters &params) {
    if (wfn1.have_energies && wfn2.have_energies) {
        occ::log::debug("Already have monomer energies, skipping");
        return;
    }

    using occ::qm::expectation;
    compute_ce_core_matrices<kind>(wfn1, proc);
    wfn2.V = wfn1.V;
    wfn2.T = wfn1.T;
    wfn2.Vecp = wfn1.Vecp;
    wfn2.H = wfn1.H;
    compute_ce_core_energies<kind>(wfn1, proc);
    compute_ce_core_energies<kind>(wfn2, proc);
    occ::log::debug("computing J with K");

    if (params.neglect_exchange) {
        occ::log::debug("neglecting K, only computing J");
        std::vector<Mat> js = proc.compute_J_list({wfn1.mo, wfn2.mo}, params.Schwarz);

        wfn1.J = js[0];
        wfn1.K = Mat::Zero(wfn1.J.rows(), wfn1.J.cols());
        wfn2.J = js[1];
        wfn2.K = Mat::Zero(wfn2.J.rows(), wfn2.J.cols());
    } else {
        occ::log::debug("computing J with K");
        std::vector<qm::JKPair> jks = proc.compute_JK_list({wfn1.mo, wfn2.mo}, params.Schwarz);

        wfn1.J = jks[0].J;
        wfn1.K = jks[0].K;

        wfn2.J = jks[1].J;
        wfn2.K = jks[1].K;
    }

    wfn1.energy.coulomb = expectation<kind>(wfn1.mo.D, wfn1.J);
    wfn1.energy.exchange = -expectation<kind>(wfn1.mo.D, wfn1.K);
    wfn2.energy.coulomb = expectation<kind>(wfn2.mo.D, wfn2.J);
    wfn2.energy.exchange = -expectation<kind>(wfn2.mo.D, wfn2.K);

    wfn1.have_energies = true;
    wfn2.have_energies = true;
}

template <SpinorbitalKind kind>
void compute_ce_model_energies(Wavefunction &wfn, HartreeFock &proc,
                               const CEMonomerCalculationParameters &params) {

    //if (wfn.have_energies) {
    //    occ::log::debug("Already have monomer energies, skipping");
    //    return;
    //}
    using occ::qm::expectation;
    using occ::qm::matrix_dimensions;
    compute_ce_core_matrices<kind>(wfn, proc);
    compute_ce_core_energies<kind>(wfn, proc);

    if (params.neglect_exchange) {
        occ::log::debug("neglecting K, only computing J");
        wfn.J = proc.compute_J(wfn.mo, params.Schwarz);
        wfn.K = Mat::Zero(wfn.J.rows(), wfn.J.cols());
    } else {
        occ::log::debug("computing J with K");
        qm::JKPair jk = proc.compute_JK(wfn.mo, params.Schwarz);
        wfn.J = jk.J;
        wfn.K = jk.K;
    }
    wfn.energy.coulomb = expectation<kind>(wfn.mo.D, wfn.J);
    wfn.energy.exchange = -expectation<kind>(wfn.mo.D, wfn.K);
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

void compute_ce_model_energies(Wavefunction &wfn, HartreeFock &hf,
                               const CEMonomerCalculationParameters &params) {
    if (wfn.is_restricted()) {
        occ::log::debug("Restricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Restricted>(
            wfn, hf, params);
    } else {
        occ::log::debug("Unrestricted wavefunction");
        compute_ce_model_energies<SpinorbitalKind::Unrestricted>(
            wfn, hf, params);
    }

    if (params.xdm) {
        compute_xdm_parameters(wfn);
    }
}

void CEModelInteraction::set_use_density_fitting(bool value) {
    m_use_density_fitting = value;
}

void CEModelInteraction::set_use_xdm_dimer_parameters(bool value) {
    m_use_xdm_dimer_parameters = value;
}

void CEModelInteraction::compute_monomer_energies(Wavefunction &wfn) const {
    HartreeFock hf(wfn.basis);
    if (m_use_density_fitting) {
        occ::log::debug("Setting DF basis: def2-universal-jkfit");
        hf.set_density_fitting_basis("def2-universal-jkfit");
    }
    CEMonomerCalculationParameters params;
    params.Schwarz = hf.compute_schwarz_ints();
    params.xdm = m_scale_factors.xdm;

    occ::log::info("Calculating xdm parameters: {}", m_scale_factors.xdm);
    if (m_scale_factors.xdm) {
        occ::log::info("XDM damping parameters: {} {}", m_scale_factors.xdm_a1,
                       m_scale_factors.xdm_a2);
    }
    compute_ce_model_energies(wfn, hf, params);
}

double population_difference(const Wavefunction &ABo, const Wavefunction &ABn,
                             const Mat &S_AB) {
    Mat Cocc_diff = ABo.mo.Cocc - ABn.mo.Cocc;
    if (ABn.is_restricted()) {
        occ::log::debug("ABo population: {}", 2 * (ABo.mo.D * S_AB).trace());
        occ::log::debug("ABn population: {}", 2 * (ABn.mo.D * S_AB).trace());
        Mat Ddiff = occ::qm::orb::density_matrix_restricted(Cocc_diff);
        return 2 * (Ddiff * S_AB).trace();

    } else {
        occ::log::debug("ABo a population: {}", 2 * (occ::qm::block::a(ABo.mo.D) * S_AB).trace());
        occ::log::debug("ABn a population: {}", 2 * (occ::qm::block::a(ABo.mo.D) * S_AB).trace());

        occ::log::debug("ABo b population: {}", 2 * (occ::qm::block::b(ABo.mo.D) * S_AB).trace());
        occ::log::debug("ABn b population: {}", 2 * (occ::qm::block::b(ABo.mo.D) * S_AB).trace());
        Mat Ddiff = occ::qm::orb::density_matrix_unrestricted(
            Cocc_diff, ABo.mo.n_alpha, ABo.mo.n_beta);
        return 2 * ((occ::qm::block::a(Ddiff) * S_AB).trace() +
                    (occ::qm::block::b(Ddiff) * S_AB).trace());
    }
}

CEEnergyComponents CEModelInteraction::operator()(Wavefunction &A,
                                                  Wavefunction &B) const {
    using occ::disp::ce_model_dispersion_energy;
    using occ::qm::Energy;

    HartreeFock hf_a(A.basis);
    HartreeFock hf_b(B.basis);
    if (m_use_density_fitting) {
        occ::log::debug("Setting DF basis for monomers: def2-universal-jkfit");
        hf_a.set_density_fitting_basis("def2-universal-jkfit");
        hf_b.set_density_fitting_basis("def2-universal-jkfit");
    }

    CEMonomerCalculationParameters params_a;
    params_a.Schwarz = hf_a.compute_schwarz_ints();
    params_a.xdm = m_scale_factors.xdm;

    CEMonomerCalculationParameters params_b;
    params_b.Schwarz = hf_b.compute_schwarz_ints();
    params_b.xdm = m_scale_factors.xdm;

    compute_ce_model_energies(A, hf_a, params_a);
    compute_ce_model_energies(B, hf_b, params_b);

    Wavefunction ABn(A, B);

    occ::log::debug("Merged wavefunction atoms:");
    for (const auto &a : ABn.atoms) {
        occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}", a.atomic_number,
                        a.x / occ::units::ANGSTROM_TO_BOHR,
                        a.y / occ::units::ANGSTROM_TO_BOHR,
                        a.z / occ::units::ANGSTROM_TO_BOHR);
    }

    // Can reuse the same HartreeFock object for both merged wfns: same
    // basis and atoms
    auto hf_AB = HartreeFock(ABn.basis);
    CEMonomerCalculationParameters params_ab;
    params_ab.Schwarz = hf_AB.compute_schwarz_ints();
    params_ab.xdm = m_scale_factors.xdm && m_use_xdm_dimer_parameters;

    if (m_use_density_fitting) {
        occ::log::debug("Setting DF basis for dimer: def2-universal-jkfit");
        hf_AB.set_density_fitting_basis("def2-universal-jkfit");
    }

    Wavefunction ABo = ABn;
    Mat S_AB = hf_AB.compute_overlap_matrix();
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();
    double p = population_difference(ABo, ABn, S_AB);

    occ::log::debug("Population difference: {:10.3g}\n", p);
    constexpr double p_tolerance = 1e-6;
    if (p < p_tolerance) {
        params_ab.neglect_exchange = true;
    }
    occ::log::debug("Num atoms A = {}\n", A.atoms.size());
    occ::log::debug("A num_electrons = {}\n", A.num_electrons);
    occ::log::debug("A nbf = {}\n", A.basis.nbf());
    occ::log::debug("A has ECPs = {}\n", A.basis.have_ecps());
    occ::log::debug("Num atoms B = {}\n", A.atoms.size());
    occ::log::debug("B num_electrons = {}\n", B.num_electrons);
    occ::log::debug("B nbf = {}\n", B.basis.nbf());
    occ::log::debug("B has ECPs = {}\n", A.basis.have_ecps());
    occ::log::debug("Num atoms AB = {}\n", ABn.atoms.size());
    occ::log::debug("AB num_electrons = {}\n", ABn.num_electrons);
    occ::log::debug("AB nbf = {}\n", ABn.basis.nbf());
    occ::log::debug("AB has ECPs = {}\n", ABn.basis.have_ecps());

    /*
    std::vector<qm::MolecularOrbitals> mos{ABn.mo, ABo.mo};
    auto jkpair = hf_AB.integral_engine().coulomb_and_exchange_list(
        ABn.mo.kind, mos, params_ab.Schwarz);
    fmt::print("J1: {}\n", jkpair[0].J.sum());
    fmt::print("K1: {}\n", jkpair[0].K.sum());
    fmt::print("J2: {}\n", jkpair[1].J.sum());
    fmt::print("K1: {}\n", jkpair[1].K.sum());
    */
    // no need to XDM for the combined wavefunctions
    //
    switch(ABn.mo.kind) {
	case SpinorbitalKind::Restricted:
	    compute_ce_model_energies_int<SpinorbitalKind::Restricted>(ABn, ABo, hf_AB, params_ab);
	    break;
	case SpinorbitalKind::Unrestricted:
	    compute_ce_model_energies_int<SpinorbitalKind::Unrestricted>(ABn, ABo, hf_AB, params_ab);
	    break;
	default:
	    throw std::runtime_error("Invalid spinorbital kind for CE model energies");
    }

    /*
    fmt::print("J1: {}\n", ABn.J.sum());
    fmt::print("K1: {}\n", ABn.K.sum());
    fmt::print("J2: {}\n", ABo.J.sum());
    fmt::print("K1: {}\n", ABo.K.sum());
    */

    occ::log::debug("ABn\n{}\n", ABn.energy.to_string());
    occ::log::debug("ABo\n{}\n", ABo.energy.to_string());

    Energy E_ABn = ABn.energy - (A.energy + B.energy);

    CEEnergyComponents energy;
    energy.coulomb = E_ABn.coulomb + E_ABn.nuclear_attraction +
                     E_ABn.nuclear_repulsion + E_ABn.ecp;
    occ::log::debug("Coulomb components:");
    occ::log::debug("A coulomb term   {:20.12f}", A.energy.coulomb);
    occ::log::debug("B coulomb term   {:20.12f}", B.energy.coulomb);
    occ::log::debug("ABn coulomb term {:20.12f}", E_ABn.coulomb);
    occ::log::debug("A en term        {:20.12f}", A.energy.nuclear_attraction);
    occ::log::debug("B en term        {:20.12f}", B.energy.nuclear_attraction);
    occ::log::debug("ABn en term      {:20.12f}", E_ABn.nuclear_attraction);
    occ::log::debug("A nn term        {:20.12f}", A.energy.nuclear_repulsion);
    occ::log::debug("B nn term        {:20.12f}", B.energy.nuclear_repulsion);
    occ::log::debug("ABn nn term      {:20.12f}", ABn.energy.nuclear_repulsion);
    occ::log::debug("ABn nn diff term {:20.12f}", E_ABn.nuclear_repulsion);
    occ::log::debug("Total term       {:20.12f}", energy.coulomb);
    double eABn = ABn.energy.core + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.coulomb;
    if (!params_ab.neglect_exchange) {
        eABn += ABn.energy.exchange;
        eABo += ABo.energy.exchange;
    } else {
        E_ABn.exchange = 0.0;
    }

    double E_rep = eABo - eABn;
    energy.repulsion = E_rep;
    energy.orthogonal_term = eABo;
    energy.nonorthogonal_term = eABn;
    energy.exchange = E_ABn.exchange;
    energy.exchange_repulsion = energy.repulsion + energy.exchange;
    occ::log::debug("Exchange repulsion components:");
    occ::log::debug("ABn core term      {:20.12f}", ABn.energy.core);
    occ::log::debug("ABn exchange term  {:20.12f}", ABn.energy.exchange);
    occ::log::debug("ABn coulomb term   {:20.12f}", ABn.energy.coulomb);
    occ::log::debug("ABo core term      {:20.12f}", ABo.energy.core);
    occ::log::debug("ABo exchange term  {:20.12f}", ABo.energy.exchange);
    occ::log::debug("ABo coulomb term   {:20.12f}", ABo.energy.coulomb);
    occ::log::debug("nonorthogonal term {:20.12f}", eABn);
    occ::log::debug("orthogonal term    {:20.12f}", eABo);
    occ::log::debug("E_rep term         {:20.12f}", E_rep);
    occ::log::debug("Exchange term      {:20.12f}", E_ABn.exchange);
    occ::log::debug("Total term         {:20.12f}", energy.exchange_repulsion);
    occ::log::debug("Test term          {:20.12f}",
                    ABo.energy.exchange - A.energy.exchange -
                        B.energy.exchange +
                        (ABo.energy.core - ABn.energy.core) +
                        (ABo.energy.coulomb - ABn.energy.coulomb));

    if (m_scale_factors.xdm) {
        occ::log::debug("XDM params: {} {}", m_scale_factors.xdm_a1,
                        m_scale_factors.xdm_a2);
        if (m_use_xdm_dimer_parameters) {
            occ::log::debug("Computing dimer parameters for XDM pair energy");
            occ::xdm::XDM xdm_calc_a(
                A.basis, A.charge(),
                {m_scale_factors.xdm_a1, m_scale_factors.xdm_a2});
            auto energy_a = xdm_calc_a.energy(A.mo);
            occ::xdm::XDM xdm_calc_b(
                B.basis, B.charge(),
                {m_scale_factors.xdm_a1, m_scale_factors.xdm_a2});
            auto energy_b = xdm_calc_b.energy(B.mo);
            occ::xdm::XDM xdm_calc_ab(
                ABn.basis, ABn.charge(),
                {m_scale_factors.xdm_a1, m_scale_factors.xdm_a2});
            auto energy_ab = xdm_calc_ab.energy(ABn.mo);
            energy.dispersion = energy_ab - energy_a - energy_b;
            A.xdm_polarizabilities = ABn.xdm_polarizabilities.block(
                0, 0, A.xdm_polarizabilities.rows(), 1);
            B.xdm_polarizabilities = ABn.xdm_polarizabilities.block(
                A.xdm_polarizabilities.rows(), 0, B.xdm_polarizabilities.rows(),
                1);
        } else {
            occ::log::debug("Using monomer parameters for XDM pair energy");
            auto xdm_result = xdm::xdm_dispersion_interaction_energy(
                {A.atoms, A.xdm_polarizabilities, A.xdm_moments, A.xdm_volumes,
                 A.xdm_free_volumes},
                {B.atoms, B.xdm_polarizabilities, B.xdm_moments, B.xdm_volumes,
                 B.xdm_free_volumes},
                {m_scale_factors.xdm_a1, m_scale_factors.xdm_a2});
            energy.dispersion = std::get<0>(xdm_result);
        }
    } else {
        energy.dispersion = ce_model_dispersion_energy(A.atoms, B.atoms);
    }
    energy.polarization = compute_polarization_energy(A, hf_a, B, hf_b);

    energy.total = m_scale_factors.scaled_total(
        energy.coulomb, energy.exchange, energy.repulsion, energy.polarization,
        energy.dispersion);
    return energy;
}

} // namespace occ::interaction
