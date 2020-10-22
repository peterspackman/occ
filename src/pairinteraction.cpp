#include "pairinteraction.h"

namespace tonto::interaction {

PairInteraction::PairInteraction(const std::shared_ptr<tonto::qm::Wavefunction>& w1,
                                 const std::shared_ptr<tonto::qm::Wavefunction>& w2) : m_wfn_a(w1), m_wfn_b(w2), m_wfn()
{
    const auto &a1 = w1.get()->atoms(), &a2 = w2.get()->atoms();
    const auto &b1 = w1.get()->basis(), &b2 = w2.get()->basis();
    std::vector<libint2::Atom> res_atoms = a1;
    res_atoms.reserve(a1.size() + a1.size());
    res_atoms.insert(res_atoms.end(), a2.begin(), a2.end());
    libint2::BasisSet res_basis = b1;
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
