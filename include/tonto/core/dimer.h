#pragma once
#include <tonto/core/molecule.h>
#include <optional>

namespace tonto::chem {
using tonto::IVec;
using tonto::Mat3N;
using tonto::Vec;
using tonto::chem::Molecule;

class Dimer{
public:
  Dimer(const Molecule&, const Molecule&);
  Dimer(const std::vector<libint2::Atom>,
        const std::vector<libint2::Atom>);

  const Molecule& a() const { return m_a; }
  const Molecule& b() const { return m_a; }

  double centroid_distance() const;
  double center_of_mass_distance() const;
  double nearest_distance() const;
  std::optional<tonto::Mat4> symmetry_relation() const;

  const Vec vdw_radii() const;
  IVec atomic_numbers() const;
  Mat3N positions() const;

  int num_electrons() const { return m_a.num_electrons() + m_b.num_electrons(); }
  int charge(int) const { return m_a.charge() + m_b.charge(); };
  int multiplicity(int) const { return m_a.multiplicity() + m_b.multiplicity() - 1; };

  bool operator ==(const Dimer&) const;

private:
  Molecule m_a, m_b;
};

} 
