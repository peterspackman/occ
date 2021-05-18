#pragma once
#include <istream>
#include <string>
#include <vector>
#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>
#include <occ/qm/spinorbital.h>

namespace occ::io {

class MoldenReader
{
public:
    MoldenReader(const std::string&);
    MoldenReader(std::istream&);

    auto spinorbital_kind() const {
        if(m_total_beta_occupation > 0) return occ::qm::SpinorbitalKind::Unrestricted;
        return occ::qm::SpinorbitalKind::Restricted;
    }
    const occ::qm::BasisSet& basis_set() const { return m_basis; }
    const std::vector<libint2::Atom>& atoms() const { return m_atoms; }
    size_t nbf() const { return m_basis.nbf(); }
    size_t num_electrons() const { return static_cast<size_t>(m_num_electrons); }

    size_t num_alpha() const {
        // note could be an issue in future, should do a better job here
        size_t n = static_cast<size_t>(m_total_alpha_occupation);
        if(m_total_alpha_occupation == m_num_electrons) {
            // beta > alpha in our convention, & this should only occur for restricted
            return n / 2;
        }
        return n;
    }

    size_t num_beta() const {
        size_t n = static_cast<size_t>(m_total_alpha_occupation);
        if(m_total_alpha_occupation == m_num_electrons) {
            // beta > alpha in our convention, & this should only occur for restricted,
            // might be important for ROHF in future
            if(n % 2 == 0) return n / 2;
            else return n / 2 + 1;
        }
        return static_cast<size_t>(m_total_beta_occupation);
    }

    const occ::MatRM& alpha_mo_coefficients() const
    {
        return m_molecular_orbitals_alpha;
    }

    const occ::MatRM& beta_mo_coefficients() const
    {
        return m_molecular_orbitals_beta;
    }

    const occ::Vec& alpha_mo_energies() const
    {
        return m_energies_alpha;
    }

    const occ::Vec& beta_mo_energies() const
    {
        return m_energies_beta;
    }



    occ::MatRM convert_mo_coefficients_from_molden_convention(const occ::qm::BasisSet&, const occ::MatRM&) const;

private:
    void parse(std::istream&);
    void parse_section(const std::string&, const std::optional<std::string>& args, std::istream&);
    void parse_atoms_section(const std::optional<std::string>&, std::istream&);
    void parse_gto_section(const std::optional<std::string>&, std::istream&);
    void parse_mo_section(const std::optional<std::string>&, std::istream&);
    void parse_mo(size_t&, size_t&, std::istream&);

    std::vector<libint2::Atom> m_atoms;
    occ::qm::BasisSet m_basis;
    std::string m_filename;
    occ::MatRM m_molecular_orbitals_alpha;
    occ::MatRM m_molecular_orbitals_beta;
    occ::Vec m_energies_alpha;
    occ::Vec m_energies_beta;
    double m_total_alpha_occupation{0};
    double m_total_beta_occupation{0};
    double m_num_electrons{0};
    bool m_pure{false};
};

}
