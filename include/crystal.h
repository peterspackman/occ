#pragma once
#include "bondgraph.h"
#include "linear_algebra.h"
#include "molecule.h"
#include "spacegroup.h"
#include "unitcell.h"
#include <vector>

namespace tonto::crystal {

using tonto::IVec;
using tonto::Mat3N;
using tonto::chem::Molecule;
using tonto::graph::PeriodicBondGraph;

struct HKL {
  int h{0}, k{0}, l{0};
};

struct AtomSlab {
  Mat3N frac_pos;
  Mat3N cart_pos;
  IVec asym_idx;
  IVec atomic_numbers;
  IVec symop;
  void resize(size_t n) {
    frac_pos.resize(3, n);
    cart_pos.resize(3, n);
    asym_idx.resize(n);
    atomic_numbers.resize(n);
    symop.resize(n);
  }
  size_t size() const { return frac_pos.cols(); }
};

struct AsymmetricUnit {
  Mat3N positions;
  IVec atomic_numbers;
  Vec occupations;
  std::vector<std::string> labels;
  std::string chemical_formula() const;
  Vec covalent_radii() const;
};

class Crystal {
public:
  Crystal(const AsymmetricUnit &, const SpaceGroup &, const UnitCell &);
  const std::vector<std::string> &labels() const {
    return m_asymmetric_unit.labels;
  }
  const Mat3N &frac() const { return m_asymmetric_unit.positions; }
  inline auto to_fractional(const Mat3N &p) const {
    return m_unit_cell.to_fractional(p);
  }
  inline auto to_cartesian(const Mat3N &p) const {
    return m_unit_cell.to_cartesian(p);
  }
  inline int num_sites() const {
    return m_asymmetric_unit.atomic_numbers.size();
  }
  inline const std::vector<SymmetryOperation> &symmetry_operations() const {
    return m_space_group.symmetry_operations();
  }
  const SpaceGroup &space_group() const { return m_space_group; }
  const AsymmetricUnit &asymmetric_unit() const { return m_asymmetric_unit; }
  const UnitCell &unit_cell() const { return m_unit_cell; }
  AtomSlab slab(const HKL &, const HKL &) const;
  const AtomSlab &unit_cell_atoms() const;
  const PeriodicBondGraph &unit_cell_connectivity() const;
  const std::vector<Molecule> &unit_cell_molecules() const;

private:
  AsymmetricUnit m_asymmetric_unit;
  SpaceGroup m_space_group;
  UnitCell m_unit_cell;
  mutable std::vector<PeriodicBondGraph::vertex_t> m_bond_graph_vertices;
  mutable PeriodicBondGraph m_bond_graph;
  mutable AtomSlab m_unit_cell_atoms;
  mutable bool m_unit_cell_atoms_needs_update{true};
  mutable bool m_unit_cell_connectivity_needs_update{true};
  mutable std::vector<Molecule> m_unit_cell_molecules{};
};

} // namespace tonto::crystal
