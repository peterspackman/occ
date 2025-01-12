#pragma once
#include <fstream>
#include <istream>
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>
#include <occ/qm/spinorbital.h>
#include <string>
#include <vector>

namespace occ::io {

class OrcaJSONReader {
public:
  using Atom = occ::core::Atom;
  OrcaJSONReader(const std::string &filename);
  OrcaJSONReader(std::istream &);

  inline auto num_basis_functions() const { return m_num_basis_functions; }
  inline auto num_orbitals() const { return m_num_basis_functions; }
  inline auto num_electrons() const { return m_num_electrons; }
  inline auto scf_energy() const { return m_scf_energy; }
  inline auto num_alpha() const { return m_num_alpha; }
  inline auto num_beta() const { return m_num_beta; }

  inline const auto &atomic_numbers() const { return m_atomic_numbers; }
  inline const auto &atom_labels() const { return m_atom_labels; }
  inline const auto &atom_positions() const { return m_atom_positions; }
  inline const auto &basis_set() const { return m_basis; }
  inline const auto &alpha_mo_energies() const { return m_alpha_energies; }
  inline const auto &alpha_mo_coefficients() const { return m_alpha_coeffs; }
  inline const auto &beta_mo_energies() const { return m_beta_energies; }
  inline const auto &beta_mo_coefficients() const { return m_beta_coeffs; }
  inline const auto &spinorbital_kind() const { return m_spinorbital_kind; }

  Mat scf_density_matrix() const;

  const Mat &overlap_matrix() const { return m_overlap; };

  std::vector<Atom> atoms() const;

private:
  void parse(std::istream &);
  void open(const std::string &filename);
  void close();

  std::ifstream m_json_file;
  size_t m_num_electrons{0};
  size_t m_num_basis_functions{0};
  size_t m_num_alpha{0};
  size_t m_num_beta{0};
  double m_scf_energy{0.0};
  std::vector<std::string> m_atom_labels;
  IVec m_atomic_numbers;
  Mat3N m_atom_positions;
  occ::qm::AOBasis m_basis;
  Mat m_alpha_coeffs, m_beta_coeffs;
  Vec m_alpha_energies, m_beta_energies;
  std::vector<std::string> m_alpha_labels, m_beta_labels;
  Mat m_overlap;
  qm::SpinorbitalKind m_spinorbital_kind{qm::SpinorbitalKind::Restricted};
};

} // namespace occ::io
