#pragma once
#include <occ/core/graph.h>

namespace occ::core {
inline double covalent_bond_tolerance = 0.4;
inline void set_bond_tolerance(float t) { covalent_bond_tolerance = t; }
inline double get_bond_tolerance() { return covalent_bond_tolerance; }

} // namespace occ::core

namespace occ::core::graph {

/**
 * Class representing and holding data for a graph edge in 3D periodic boundary
 * conditions.
 */
struct PeriodicEdge {
  double dist{0.0};
  size_t source{0}, target{0};
  size_t source_asym_idx{0}, target_asym_idx{0};
  int h{0}, k{0}, l{0};
};

/**
 * Class representing and holding data for a graph vertex in 3D periodic
 * boundary conditions.
 */
struct PeriodicVertex {
  size_t uc_idx{0};
};

using PeriodicBondGraph = Graph<PeriodicVertex, PeriodicEdge>;
} // namespace occ::core::graph
