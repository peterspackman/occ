#include "pairinteraction.h"

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

PairInteraction::PairInteraction(const std::shared_ptr<tonto::qm::Wavefunction>& w1,
                                 const std::shared_ptr<tonto::qm::Wavefunction>& w2) : m_wfn_a(w1), m_wfn_b(w2), m_wfn()
{
    const auto &a1 = w1.get()->atoms(), &a2 = w2.get()->atoms();
    const auto &b1 = w1.get()->basis(), &b2 = w2.get()->basis();
    std::vector<libint2::Atom> res_atoms = a1;
    res_atoms.reserve(a1.size() + a1.size());
    res_atoms.insert(res_atoms.end(), a2.begin(), a2.end());
    BasisSet res_basis = b1;
    res_basis.reserve(b1.size() + b2.size());
    res_basis.insert(res_basis.end(), b2.begin(), b2.end());
    m_wfn = tonto::qm::Wavefunction(res_basis, res_atoms);
}

void PairInteraction::merge_molecular_orbitals()
{
    const auto &c1 = m_wfn_a.get()->molecular_orbitals(), &c2 = m_wfn_b.get()->molecular_orbitals();
    size_t nbf = c1.rows() + c2.rows();
    MatRM result = MatRM::Zero(nbf, nbf);
    result.topLeftCorner(c1.rows(), c1.cols()) = c1;
    result.bottomRightCorner(c2.rows(), c2.cols()) = c2;

    tonto::ints::shellpair_list_t shellpair_list;
    tonto::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = tonto::ints::compute_shellpairs(m_wfn.basis());
    MatRM S = tonto::ints::compute_1body_ints<libint2::Operator::overlap>(m_wfn.basis(), shellpair_list)[0];
    double condition_threshold = 1.0 / std::numeric_limits<double>::epsilon();
    MatRM X, Xinv;
    double XtX_condition_number;
    std::tie(X, Xinv, XtX_condition_number) = conditioning_orthogonalizer(S, condition_threshold);
    m_wfn.set_molecular_orbitals(X * result * Xinv);

}

}
