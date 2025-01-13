#pragma once
#include <occ/cg/neighbor_atoms.h>
#include <occ/cg/solvation_contribution.h>
#include <occ/cg/solvent_surface.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <vector>

namespace occ::cg {

struct NeighborContribution {
  double coulomb{0.0};
  double cds{0.0};
  double area_coulomb{0.0};
  double area_cds{0.0};
  bool neighbor_set{false};

  void assign(const NeighborContribution &other);
};

enum class PartitionScheme {
  NearestAtom,
  NearestAtomDnorm,
  ElectronDensity,
};

} // namespace occ::cg
