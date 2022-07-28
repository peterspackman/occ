#pragma once
#include <istream>
#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <optional>
#include <string>
#include <vector>

namespace occ::io {

class MoldenReader {
  public:
    enum class Source {
        Unknown,
        Orca,
        NWChem,
    };

    MoldenReader(const std::string &);
    MoldenReader(std::istream &);

    auto spinorbital_kind() const {
        if (m_total_beta_occupation > 0)
            return occ::qm::SpinorbitalKind::Unrestricted;
        return occ::qm::SpinorbitalKind::Restricted;
    }
    const std::vector<occ::core::Atom> &atoms() const { return m_atoms; }
    occ::qm::AOBasis basis_set() const {
        return occ::qm::AOBasis(atoms(), m_shells);
    }
    size_t nbf() const { return basis_set().nbf(); }
    size_t num_electrons() const {
        return static_cast<size_t>(m_num_electrons);
    }

    size_t num_alpha() const {
        // note could be an issue in future, should do a better job here
        size_t n = static_cast<size_t>(m_total_alpha_occupation);
        if (m_total_alpha_occupation == m_num_electrons) {
            // beta > alpha in our convention, & this should only occur for
            // restricted
            return n / 2;
        }
        return n;
    }

    size_t num_beta() const {
        size_t n = static_cast<size_t>(m_total_alpha_occupation);
        if (m_total_alpha_occupation == m_num_electrons) {
            // beta > alpha in our convention, & this should only occur for
            // restricted, might be important for ROHF in future
            if (n % 2 == 0)
                return n / 2;
            else
                return n / 2 + 1;
        }
        return static_cast<size_t>(m_total_beta_occupation);
    }

    const Mat &alpha_mo_coefficients() const {
        return m_molecular_orbitals_alpha;
    }

    const Mat &beta_mo_coefficients() const {
        return m_molecular_orbitals_beta;
    }

    const occ::Vec &alpha_mo_energies() const { return m_energies_alpha; }

    const occ::Vec &beta_mo_energies() const { return m_energies_beta; }

    Mat convert_mo_coefficients_from_molden_convention(const occ::qm::AOBasis &,
                                                       const Mat &) const;

  private:
    void parse(std::istream &);
    void parse_section(const std::string &,
                       const std::optional<std::string> &args, std::istream &);
    void parse_atoms_section(const std::optional<std::string> &,
                             std::istream &);
    void parse_gto_section(const std::optional<std::string> &, std::istream &);
    void parse_mo_section(const std::optional<std::string> &, std::istream &);
    void parse_title_section(const std::optional<std::string> &,
                             std::istream &);
    void parse_mo(size_t &, size_t &, std::istream &);

    std::vector<occ::core::Atom> m_atoms;
    std::vector<occ::qm::Shell> m_shells;
    std::string m_filename;
    Mat m_molecular_orbitals_alpha;
    Mat m_molecular_orbitals_beta;
    occ::Vec m_energies_alpha;
    occ::Vec m_energies_beta;
    double m_total_alpha_occupation{0};
    double m_total_beta_occupation{0};
    double m_num_electrons{0};
    std::string m_current_line;
    bool m_pure{false};
    Source source{Source::Unknown};
};

} // namespace occ::io
