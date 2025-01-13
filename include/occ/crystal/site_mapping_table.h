#pragma once
#include <occ/core/graph.h>
#include <occ/crystal/site_index.h>
#include <vector>

namespace occ::crystal {

class Crystal;

struct SiteMappingEdge {
  size_t source{0}, target{0};
  int symop{16484};
  HKL offset;
};

struct SiteMappingVertex {
  size_t index{0};
};

using SiteMappingGraph = core::graph::Graph<SiteMappingVertex, SiteMappingEdge>;

class SiteMappingTable {
public:
  // Get all edges (symmetry operations) connecting two vertices
  std::vector<SiteMappingEdge> get_edges(size_t source, size_t target) const;

  // Get target vertex given a source and a symmetry operation
  std::optional<size_t> get_target(size_t source, int symop,
                                   const HKL &offset) const;

  // Get target vertex given a source and a symmetry operation
  std::optional<std::pair<size_t, HKL>> get_target_and_offset(size_t source,
                                                              int symop) const;

  // Get all neighbors of a vertex
  std::vector<size_t> get_neighbors(size_t source) const;

  // Get all symmetry operations (with offsets) that can be applied to a vertex
  std::vector<std::pair<int, HKL>> get_symmetry_operations(size_t source) const;

  // Get the number of vertices in the graph
  size_t size() const;

  static SiteMappingTable build_atom_table(const Crystal &);
  static SiteMappingTable build_molecule_table(const Crystal &);

private:
  SiteMappingGraph m_graph;
};

} // namespace occ::crystal
