#pragma once
#include "linear_algebra.h"
#include "basisset.h"

namespace tonto::qm {

class Wavefunction {
public:
    Wavefunction() {}
    Wavefunction(const BasisSet& basis, const std::vector<libint2::Atom>& atoms) :
        m_basis(basis), m_atoms(atoms)
    {}

    const tonto::MatRM& molecular_orbitals() const { return m_C; }
    const auto& basis() const { return m_basis; }
    const auto& atoms() const { return m_atoms; }
    void set_molecular_orbitals(const tonto::MatRM& c) { m_C = c; }

private:
    BasisSet m_basis;
    std::vector<libint2::Atom> m_atoms;
    tonto::MatRM m_C, m_Cocc;
    tonto::Vec m_orbital_energies;
    tonto::MatRM m_kinetic_matrix;
    tonto::MatRM m_nuclear_attraction_matrix;
    tonto::MatRM m_fock_matrix;
    tonto::MatRM m_core_hamiltonian;
    tonto::MatRM m_coulomb_matrix;
    tonto::MatRM m_exchange_matrix;
    tonto::MatRM m_overlap_matrix;
};

}
