#pragma once
#include "ints.h"
#include "wavefunction.h"
#include "basisset.h"
#include "spinorbital.h"
#include <memory>

namespace tonto::interaction {
using tonto::MatRM;
using tonto::Vec;
using tonto::qm::BasisSet;
using tonto::qm::SpinorbitalKind;

struct CEModelEnergyScaleFactors {
    double coulomb{1.0};
    double exchange_repulsion{1.0};
    double polarization{1.0};
    double dispersion{1.0};
    double scaled_total(double coul, double ex, double pol, double disp)
    {
        return coulomb * coul + exchange_repulsion * ex + polarization * pol + dispersion * disp;
    }
};

inline CEModelEnergyScaleFactors CE_HF_321G{1.019, 0.811, 0.651, 0.901};
inline CEModelEnergyScaleFactors CE_B3LYP_631Gdp{1.0573, 0.6177, 0.7399, 0.8708};

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
