#pragma once
#include <istream>
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/qm/spinorbital.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace occ::io {

/**
 * Reader for Molden format wavefunction files.
 *
 * Section names and the angular-function flags ([5D], [5D7F], [5D10F], [7F],
 * [9G] and their Cartesian counterparts) are case-insensitive. After reading,
 * MO coefficients are in occ's internal AO ordering and normalization: a
 * file that mixes spherical and Cartesian shells (e.g. [5D10F]) is converted
 * to a Cartesian basis.
 *
 * Restricted open-shell files (alpha orbitals only, with singly occupied
 * orbitals) are read as unrestricted so the spin density is kept.
 */
class MoldenReader {
public:
  enum class Source {
    Unknown,
    Orca,
    NWChem,
  };

  MoldenReader(const std::string &);
  MoldenReader(std::istream &);

  occ::qm::SpinorbitalKind spinorbital_kind() const { return m_kind; }
  const std::vector<occ::core::Atom> &atoms() const { return m_atoms; }
  occ::gto::AOBasis basis_set() const {
    return occ::gto::AOBasis(atoms(), m_shells);
  }
  size_t nbf() const { return m_nbf; }
  size_t num_electrons() const { return m_num_alpha + m_num_beta; }
  size_t num_alpha() const { return m_num_alpha; }
  size_t num_beta() const { return m_num_beta; }

  // nbf x nbf, columns beyond the number of MOs in the file are zero
  const Mat &alpha_mo_coefficients() const {
    return m_molecular_orbitals_alpha;
  }
  const Mat &beta_mo_coefficients() const { return m_molecular_orbitals_beta; }

  // occupations per spin orbital: 0..1 for unrestricted, 0..2 for restricted
  const Vec &alpha_occupations() const { return m_occupations_alpha; }
  const Vec &beta_occupations() const { return m_occupations_beta; }

  const Vec &alpha_mo_energies() const { return m_energies_alpha; }
  const Vec &beta_mo_energies() const { return m_energies_beta; }

  Source source() const { return m_source; }

private:
  struct Orbital {
    double energy{0.0};
    double occupation{0.0};
    bool alpha{true};
    std::vector<std::pair<int, double>> coefficients;
  };

  using Lines = std::vector<std::string>;

  void parse(std::istream &);
  void parse_atoms_section(const std::string &args, const Lines &, size_t &);
  void parse_gto_section(const Lines &, size_t &);
  void parse_mo_section(const Lines &, size_t &);
  void parse_title_section(const Lines &, size_t &);
  void set_angular_flags(const std::string &section_name);
  void finalize();

  bool shell_is_pure_in_file(int l) const;
  Mat to_internal_convention(const Mat &file_mos, bool target_pure) const;

  std::vector<occ::core::Atom> m_atoms;
  std::vector<occ::gto::Shell> m_shells;
  std::vector<Orbital> m_orbitals;
  std::string m_filename;

  Mat m_molecular_orbitals_alpha;
  Mat m_molecular_orbitals_beta;
  occ::Vec m_energies_alpha;
  occ::Vec m_energies_beta;
  occ::Vec m_occupations_alpha;
  occ::Vec m_occupations_beta;
  occ::qm::SpinorbitalKind m_kind{occ::qm::SpinorbitalKind::Restricted};
  size_t m_nbf{0};
  size_t m_num_alpha{0};
  size_t m_num_beta{0};

  // Molden defaults to Cartesian functions for every l
  bool m_pure_d{false};
  bool m_pure_f{false};
  bool m_pure_g{false};
  bool m_have_ecp_core{false};
  Source m_source{Source::Unknown};
};

} // namespace occ::io
