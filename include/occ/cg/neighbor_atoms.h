#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
namespace occ::cg {

struct NeighborAtoms {
  NeighborAtoms(const crystal::CrystalDimers::MoleculeNeighbors &);
  Mat3N positions;
  IVec molecule_index;
  IVec atomic_numbers;
  Vec vdw_radii;
};

} // namespace occ::cg
