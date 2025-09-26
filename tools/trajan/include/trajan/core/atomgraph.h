#pragma once
#include <occ/core/atom.h>
#include <occ/core/graph.h>

namespace trajan::core {

struct BondEdge {
  double bond_length{0.0};
  std::pair<int, int> indices{
      0, 0}; // NOTE: Obtained from Atom ordering. Not their indices, which are
             // taken from the user and can therefore be anything in any order.
};

struct AtomVertex {
  int index;
};

using AtomGraph = occ::core::graph::Graph<AtomVertex, BondEdge>;
} // namespace trajan::core
