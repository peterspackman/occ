#include "hf.h"
#include "parallel.h"
#include "fock.h"

namespace tonto::hf {

HartreeFock::HartreeFock(const std::vector<libint2::Atom> &atoms,
                         const BasisSet &basis)
    : m_atoms(atoms), m_basis(basis) {
  std::tie(m_shellpair_list, m_shellpair_data) = tonto::ints::compute_shellpairs(m_basis);
  for (const auto &a : m_atoms) {
    m_num_e += a.atomic_number;
  }
  m_num_e -= m_charge;
}

double HartreeFock::nuclear_repulsion_energy() const {
  double enuc = 0.0;
  for (auto i = 0; i < m_atoms.size(); i++)
    for (auto j = i + 1; j < m_atoms.size(); j++) {
      auto xij = m_atoms[i].x - m_atoms[j].x;
      auto yij = m_atoms[i].y - m_atoms[j].y;
      auto zij = m_atoms[i].z - m_atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      enuc += m_atoms[i].atomic_number * m_atoms[j].atomic_number / r;
    }
  return enuc;
}

MatRM HartreeFock::compute_shellblock_norm(const MatRM &A) const {
  return tonto::ints::compute_shellblock_norm(m_basis, A);
}

MatRM HartreeFock::compute_2body_fock(const MatRM &D, double precision,
                                      const MatRM &Schwarz) const {
  auto F = tonto::ints::compute_fock<tonto::ints::SpinorbitalKind::Restricted>(
      m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
  m_e_alpha = D.cwiseProduct(F).sum();
  return F;
}

std::pair<MatRM, MatRM> HartreeFock::compute_JK(const MatRM &D,
                                                double precision,
                                                const MatRM &Schwarz) const {
  return tonto::ints::compute_JK(m_basis, m_shellpair_list, m_shellpair_data, D,
                                 precision, Schwarz);
}

MatRM HartreeFock::compute_J(const MatRM &D, double precision,
                             const MatRM &Schwarz) const {
  return tonto::ints::compute_J(
      m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
}

std::pair<MatRM, MatRM>
HartreeFock::compute_2body_fock_unrestricted(const MatRM &Da, const MatRM &Db,
                                             double precision,
                                             const MatRM &Schwarz) const {
  return tonto::ints::compute_2body_fock_unrestricted(
      m_basis, m_shellpair_list, m_shellpair_data, Da, Db, precision, Schwarz);
}

std::tuple<MatRM, MatRM, MatRM, MatRM>
HartreeFock::compute_JK_unrestricted(const MatRM &Da, const MatRM &Db,
                                     double precision,
                                     const MatRM &Schwarz) const {
  return tonto::ints::compute_JK_unrestricted(
      m_basis, m_shellpair_list, m_shellpair_data, Da, Db, precision, Schwarz);
}


MatRM HartreeFock::compute_2body_fock_general(const MatRM &D, double precision,
                                      const MatRM &Schwarz) const {
  return tonto::ints::compute_fock<tonto::ints::SpinorbitalKind::General>(
      m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
}

std::pair<MatRM, MatRM> HartreeFock::compute_JK_general(const MatRM &D,
                                                double precision,
                                                const MatRM &Schwarz) const {
  return tonto::ints::compute_JK_general(m_basis, m_shellpair_list, m_shellpair_data, D,
                                 precision, Schwarz);
}

Mat3N HartreeFock::nuclear_electric_field_contribution(const Mat3N &positions)
{
    Mat3N result = Mat3N::Zero(3, positions.cols());
    for(const auto& atom: m_atoms)
    {
        double Z = atom.atomic_number;
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        auto r3 = r.array() * r.array() * r.array();
        result.array() += (Z * (ab.array().rowwise() / r3));
    }
    return result;
}

} // namespace tonto::hf
