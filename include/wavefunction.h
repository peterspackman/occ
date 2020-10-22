#pragma once
#include "linear_algebra.h"
#include <libint2/basis.h>
#include <libint2/atom.h>

namespace tonto::qm {

class Wavefunction {
public:
    struct Energies
    {
        double coulomb{0.0};
        double exchange_repulsion{0.0};
        double nuclear_repulsion{0.0};
        double kinetic{0.0};
        double one_electron{0.0};
    };

    Wavefunction() {}
    Wavefunction(const libint2::BasisSet& basis, const std::vector<libint2::Atom>& atoms) :
        m_basis(basis), m_atoms(atoms)
    {}

    const tonto::MatRM& molecular_orbitals() const { return m_molecular_orbitals; }
    const auto& basis() const { return m_basis; }
    const auto& atoms() const { return m_atoms; }
    void set_molecular_orbitals(const tonto::MatRM& c) { m_molecular_orbitals = c; }

private:
    libint2::BasisSet m_basis;
    std::vector<libint2::Atom> m_atoms;
    tonto::MatRM m_molecular_orbitals;

    std::optional<Energies> m_energies;
};

}
