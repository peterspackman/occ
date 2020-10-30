#pragma once
#include "ints.h"
#include "wavefunction.h"
#include "basisset.h"
#include <memory>

namespace tonto::interaction {
using tonto::MatRM;
using tonto::Vec;
using tonto::qm::BasisSet;

std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM&, const MatRM&, const Vec&, const Vec&);
BasisSet merge_basis_sets(const BasisSet&, const BasisSet&);
std::vector<libint2::Atom> merge_atoms(const std::vector<libint2::Atom>&, const std::vector<libint2::Atom>&);

class PairInteraction
{
public:
    PairInteraction(const std::shared_ptr<tonto::qm::Wavefunction>& w1,
                    const std::shared_ptr<tonto::qm::Wavefunction>& w2);

private:
    void merge_molecular_orbitals();
    std::shared_ptr<tonto::qm::Wavefunction> m_wfn_a, m_wfn_b;
    tonto::qm::Wavefunction m_wfn;
};

}
