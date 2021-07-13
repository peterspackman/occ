#pragma once
#include <occ/core/molecule.h>
#include <optional>

namespace occ::chem {
using occ::IVec;
using occ::Mat3N;
using occ::Vec;
using occ::chem::Molecule;

class Dimer{
public:
  Dimer(const Molecule&, const Molecule&);
  Dimer(const std::vector<occ::core::Atom>,
        const std::vector<occ::core::Atom>);

  const Molecule& a() const { return m_a; }
  const Molecule& b() const { return m_b; }

  double centroid_distance() const;
  double center_of_mass_distance() const;
  double nearest_distance() const;
  std::optional<occ::Mat4> symmetry_relation() const;
  Vec3 v_ab() const;

  const Vec vdw_radii() const;
  IVec atomic_numbers() const;
  Mat3N positions() const;

  int num_electrons() const { return m_a.num_electrons() + m_b.num_electrons(); }
  int charge(int) const { return m_a.charge() + m_b.charge(); };
  int multiplicity(int) const { return m_a.multiplicity() + m_b.multiplicity() - 1; };

  bool same_asymmetric_molecule_idxs(const Dimer&) const;
  bool operator ==(const Dimer&) const;

private:
  Molecule m_a, m_b;
};

} 
