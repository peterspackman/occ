#include "pairinteraction.h"
#include "hf.h"
#include "disp.h"

namespace tonto::interaction {


std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM& mo_a, const MatRM& mo_b, const Vec& e_a, const Vec& e_b)
{
    MatRM merged = MatRM::Zero(mo_a.rows() + mo_b.rows(), mo_a.cols() + mo_b.cols());
    Vec merged_energies(e_a.rows() + e_b.rows());
    merged_energies.topRows(e_a.rows()) = e_a;
    merged_energies.bottomRows(e_b.rows()) = e_b;
    std::vector<Eigen::Index> idxs;
    idxs.reserve(merged_energies.rows());
    for(Eigen::Index i = 0; i < merged_energies.rows(); i++) idxs.push_back(i);
    std::sort(idxs.begin(), idxs.end(), [&merged_energies](Eigen::Index a, Eigen::Index b) { return merged_energies(a) < merged_energies(b); });
    Vec sorted_energies(merged_energies.rows());
    for(Eigen::Index i = 0; i < merged_energies.rows(); i++) {
        Eigen::Index c = idxs[i];
        sorted_energies(i) = merged_energies(c);
        if(c >= mo_a.cols()) {
            merged.col(i).bottomRows(mo_b.rows()) = mo_b.col(c - mo_a.cols());
        }
        else {
            merged.col(i).topRows(mo_a.rows()) = mo_a.col(c);
        }
    }
    return {merged, sorted_energies};
}


BasisSet merge_basis_sets(const BasisSet& basis_a, const BasisSet& basis_b)
{
    tonto::qm::BasisSet merged = basis_a;
    merged.reserve(basis_a.size() + basis_b.size());
    merged.insert(merged.end(), basis_b.begin(), basis_b.end());
    merged.update();
    merged.set_pure(false);
    return merged;
}

std::vector<libint2::Atom> merge_atoms(const std::vector<libint2::Atom>& atoms_a, const std::vector<libint2::Atom>& atoms_b)
{
    std::vector<libint2::Atom> merged = atoms_a;
    merged.reserve(atoms_a.size() + atoms_b.size());
    merged.insert(merged.end(), atoms_b.begin(), atoms_b.end());
    return merged;
}

CEModelInteraction::CEModelInteraction(const  CEModelEnergyScaleFactors &facs) : scale_factors(facs)
{

}


template<SpinorbitalKind kind>
void compute_ce_model_energies(Wavefunction& wfn, tonto::hf::HartreeFock& hf)
{
    using tonto::qm::expectation;
    using tonto::qm::matrix_dimensions;
    if constexpr(kind == SpinorbitalKind::Restricted) {
        wfn.V = hf.compute_nuclear_attraction_matrix();
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.D, wfn.V);
        wfn.T = hf.compute_kinetic_matrix();
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.D, wfn.T);
        wfn.H = wfn.V + wfn.T;
        wfn.energy.core = 2 * expectation<kind>(wfn.D, wfn.H);
        std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.D);
        wfn.energy.coulomb = expectation<kind>(wfn.D, wfn.J);
        wfn.energy.exchange = - expectation<kind>(wfn.D, wfn.K);
        wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
    }
    else {
        size_t rows, cols;
        std::tie(rows, cols) = matrix_dimensions<SpinorbitalKind::Unrestricted>(wfn.nbf);
        wfn.T = MatRM(rows, cols);
        wfn.V = MatRM(rows, cols);
        wfn.T.alpha() = hf.compute_kinetic_matrix();
        wfn.T.beta() = wfn.T.alpha();
        wfn.V.alpha() = hf.compute_nuclear_attraction_matrix();
        wfn.V.beta() = wfn.V.alpha();
        wfn.H = wfn.V + wfn.T;
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.D, wfn.V);
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.D, wfn.T);
        wfn.energy.core = 2 * expectation<kind>(wfn.D, wfn.H);
        std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.D);
        wfn.energy.coulomb = expectation<kind>(wfn.D, wfn.J);
        wfn.energy.exchange = - expectation<kind>(wfn.D, wfn.K);
        wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
    }

}


void compute_ce_model_energies(Wavefunction &wfn, tonto::hf::HartreeFock &hf)
{
    if(wfn.is_restricted()) return compute_ce_model_energies<SpinorbitalKind::Restricted>(wfn, hf);
    else return compute_ce_model_energies<SpinorbitalKind::Unrestricted>(wfn, hf);
}

CEModelInteraction::EnergyComponents CEModelInteraction::operator()(Wavefunction &A, Wavefunction &B) const
{
    using tonto::hf::HartreeFock;
    using tonto::qm::Energy;
    using tonto::disp::ce_model_dispersion_energy;

    HartreeFock hf_a(A.atoms, A.basis);
    HartreeFock hf_b(B.atoms, B.basis);

    compute_ce_model_energies(A, hf_a);
    compute_ce_model_energies(B, hf_b);

    Wavefunction ABn(A, B);

    // Can reuse the same HartreeFock object for both merged wfns: same basis and atoms
    auto hf_AB = HartreeFock(ABn.atoms, ABn.basis);

    Wavefunction ABo = ABn;
    MatRM S_AB = hf_AB.compute_overlap_matrix();
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();

    compute_ce_model_energies(ABn, hf_AB);
    compute_ce_model_energies(ABo, hf_AB);


    Energy E_ABn;
    E_ABn.kinetic = ABn.energy.kinetic - (A.energy.kinetic + B.energy.kinetic);
    E_ABn.coulomb = ABn.energy.coulomb - (A.energy.coulomb + B.energy.coulomb);
    E_ABn.exchange = ABn.energy.exchange - (A.energy.exchange + B.energy.exchange);
    E_ABn.core = ABn.energy.core - (A.energy.core + B.energy.core);
    E_ABn.nuclear_attraction = ABn.energy.nuclear_attraction - (A.energy.nuclear_attraction + B.energy.nuclear_attraction);
    E_ABn.nuclear_repulsion = ABn.energy.nuclear_repulsion - (A.energy.nuclear_repulsion + B.energy.nuclear_repulsion);

    EnergyComponents energy;
    energy.coulomb = E_ABn.coulomb + E_ABn.nuclear_attraction + E_ABn.nuclear_repulsion;
    double eABn = ABn.energy.core + ABn.energy.exchange + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.exchange + ABo.energy.coulomb;
    double E_rep = eABo - eABn;
    energy.exchange_repulsion = E_ABn.exchange + E_rep;
    energy.dispersion = ce_model_dispersion_energy(A.atoms, B.atoms);
    energy.polarization = compute_polarization_energy(A, hf_a, B, hf_b);

    energy.total = scale_factors.scaled_total(energy.coulomb, energy.exchange_repulsion, energy.polarization, energy.dispersion);
    return energy;
}


}
