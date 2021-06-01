#include <occ/core/parallel.h>
#include <occ/qm/hf.h>
#include <occ/qm/fock.h>
#include <occ/qm/property_ints.h>

namespace occ::hf {

HartreeFock::HartreeFock(const std::vector<libint2::Atom> &atoms,
                         const BasisSet &basis)
    : m_atoms(atoms), m_basis(basis) {
  std::tie(m_shellpair_list, m_shellpair_data) = occ::ints::compute_shellpairs(m_basis);
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
  return occ::ints::compute_shellblock_norm(m_basis, A);
}

Mat3N HartreeFock::nuclear_electric_field_contribution(const Mat3N &positions) const
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

Mat3N HartreeFock::electronic_electric_field_contribution(SpinorbitalKind kind, const MatRM& D, const Mat3N &positions) const
{
    constexpr bool use_finite_differences = true;
    if constexpr(use_finite_differences) {
        double delta = 1e-8;
        occ::Mat3N efield_fd(positions.rows(), positions.cols());
        for(size_t i = 0; i < 3; i++) {
            auto pts_delta = positions;
            pts_delta.row(i).array() += delta;
            auto esp_f = electronic_electric_potential_contribution(kind, D, pts_delta);
            pts_delta.row(i).array() -= 2 * delta;
            auto esp_b = electronic_electric_potential_contribution(kind, D, pts_delta);
            efield_fd.row(i) = - (esp_f - esp_b) / (2 * delta);
        }
        return efield_fd;
    }
    else {
        switch(kind)
        {
            case SpinorbitalKind::Restricted:
                return occ::ints::compute_electric_field<SpinorbitalKind::Restricted>(D, m_basis, m_shellpair_list, positions);
            case SpinorbitalKind::Unrestricted:
                return occ::ints::compute_electric_field<SpinorbitalKind::Unrestricted>(D, m_basis, m_shellpair_list, positions);
            case SpinorbitalKind::General:
                return occ::ints::compute_electric_field<SpinorbitalKind::General>(D, m_basis, m_shellpair_list, positions);
        }
    }
}

Vec HartreeFock::electronic_electric_potential_contribution(SpinorbitalKind kind, const MatRM &D, const Mat3N &positions) const
{
    switch(kind)
    {
        case SpinorbitalKind::Unrestricted: return occ::ints::compute_electric_potential<SpinorbitalKind::Unrestricted>(D, m_basis, m_shellpair_list, positions);
        case SpinorbitalKind::General: return occ::ints::compute_electric_potential<SpinorbitalKind::General>(D, m_basis, m_shellpair_list, positions);
        default: return occ::ints::compute_electric_potential<SpinorbitalKind::Restricted>(D, m_basis, m_shellpair_list, positions);
    }
}

Vec HartreeFock::nuclear_electric_potential_contribution(const Mat3N &positions) const
{
    Vec result = Vec::Zero(positions.cols());
    for(const auto& atom: m_atoms)
    {
        double Z = atom.atomic_number;
        Vec3 atom_pos{atom.x, atom.y, atom.z};
        auto ab = positions.colwise() - atom_pos;
        auto r = ab.colwise().norm();
        result.array() += Z / r.array();
    }
    return result;
}

} // namespace occ::hf
