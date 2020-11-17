#pragma once
#include <istream>
#include <string>
#include <vector>
#include "linear_algebra.h"
#include "basisset.h"
#include "spinorbital.h"
#include "linear_algebra.h"

namespace tonto::io {

class MoldenReader
{
public:
    MoldenReader(const std::string&);
    MoldenReader(std::istream&);
    const auto atoms() const { return m_atoms; }
    const size_t nbf() const { return m_basis.nbf(); }

private:
    void parse(std::istream&);
    void parse_section(const std::string&, const std::optional<std::string>& args, std::istream&);
    void parse_atoms_section(const std::optional<std::string>&, std::istream&);
    void parse_gto_section(const std::optional<std::string>&, std::istream&);
    void parse_mo_section(const std::optional<std::string>&, std::istream&);
    void parse_mo(size_t&, size_t&, std::istream&);

    std::vector<libint2::Atom> m_atoms;
    tonto::qm::BasisSet m_basis;
    std::string m_filename;
    tonto::MatRM m_molecular_orbitals_alpha;
    tonto::MatRM m_molecular_orbitals_beta;
    tonto::Vec m_energies_alpha;
    tonto::Vec m_energies_beta;
    size_t m_num_alpha{0};
    size_t m_num_beta{0};
};

}
