#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/qm/scf_method.h>

namespace occ::qm {

SCFMethodBase::SCFMethodBase(const std::vector<core::Atom> &atoms)
    : m_atoms(atoms), m_frozen_electrons(atoms.size(), 0) {}

Vec3 SCFMethodBase::center_of_mass() const {
  auto mol = occ::core::Molecule(m_atoms);
  return mol.center_of_mass() * occ::units::ANGSTROM_TO_BOHR;
}

void SCFMethodBase::set_system_charge(int charge) {
  m_charge = charge;
  update_electron_count();
}

void SCFMethodBase::set_frozen_electrons(const std::vector<int> &frozen) {
  m_frozen_electrons = frozen;
}

double SCFMethodBase::nuclear_repulsion_energy() const {
  double enuc = 0.0;
  for (auto i = 0; i < m_atoms.size(); i++)
    for (auto j = i + 1; j < m_atoms.size(); j++) {
      auto xij = m_atoms[i].x - m_atoms[j].x;
      auto yij = m_atoms[i].y - m_atoms[j].y;
      auto zij = m_atoms[i].z - m_atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      enuc += (m_atoms[i].atomic_number - m_frozen_electrons[i]) *
              (m_atoms[j].atomic_number - m_frozen_electrons[j]) / r;
    }
  return enuc;
}

Mat3N SCFMethodBase::nuclear_repulsion_gradient() const {
  Mat3N grad = Mat3N::Zero(3, m_atoms.size());
  for (auto i = 0; i < m_atoms.size(); i++)
    for (auto j = i + 1; j < m_atoms.size(); j++) {
      auto xij = m_atoms[i].x - m_atoms[j].x;
      auto yij = m_atoms[i].y - m_atoms[j].y;
      auto zij = m_atoms[i].z - m_atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      double fac = (m_atoms[i].atomic_number - m_frozen_electrons[i]) *
                   (m_atoms[j].atomic_number - m_frozen_electrons[j]) /
                   (r * r2);
      grad(0, i) -= fac * xij;
      grad(1, i) -= fac * yij;
      grad(2, i) -= fac * zij;

      grad(0, j) += fac * xij;
      grad(1, j) += fac * yij;
      grad(2, j) += fac * zij;
    }
  return grad;
}

Vec SCFMethodBase::nuclear_electric_potential_contribution(
    const Mat3N &positions) const {
  Vec result = Vec::Zero(positions.cols());
  int atom_index = 0;
  for (const auto &atom : m_atoms) {
    double Z = atom.atomic_number - m_frozen_electrons[atom_index];
    Vec3 atom_pos{atom.x, atom.y, atom.z};
    auto ab = positions.colwise() - atom_pos;
    auto r = ab.colwise().norm();
    result.array() += Z / r.array();
    atom_index++;
  }
  return result;
}

Mat3N SCFMethodBase::nuclear_electric_field_contribution(
    const Mat3N &positions) const {
  Mat3N result = Mat3N::Zero(3, positions.cols());
  int atom_index = 0;
  for (const auto &atom : m_atoms) {
    double Z = atom.atomic_number - m_frozen_electrons[atom_index];
    Vec3 atom_pos{atom.x, atom.y, atom.z};
    auto ab = positions.colwise() - atom_pos;
    auto r = ab.colwise().norm();
    auto r3 = r.array() * r.array() * r.array();
    result.array() += (Z * (ab.array().rowwise() / r3));
    atom_index++;
  }
  return result;
}

} // namespace occ::qm
