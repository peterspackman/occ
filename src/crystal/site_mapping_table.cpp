#include <occ/crystal/crystal.h>
#include <occ/crystal/site_mapping_table.h>

namespace occ::crystal {

SiteMappingTable SiteMappingTable::build_atom_table(const Crystal &crystal) {
  SiteMappingTable table;
  const auto &atoms = crystal.unit_cell_atoms();
  const auto &symops = crystal.symmetry_operations();

  // Add vertices
  for (size_t i = 0; i < atoms.size(); ++i) {
    table.m_graph.add_vertex({i});
  }

  // Pre-compute the tolerance for position comparison
  const double tolerance = 1e-6;

  // Iterate over symmetry operations
  for (const auto &symop : symops) {

    // Apply symmetry operation to all fractional positions at once
    Mat3N transformed_pos = symop.apply(atoms.frac_pos);

    // Compute offsets and wrap positions back to unit cell
    IMat3N offsets = transformed_pos.array().floor().cast<int>();
    transformed_pos -= offsets.cast<double>();

    // Add edges
    for (size_t i = 0; i < atoms.size(); ++i) {
      for (size_t j = 0; j < atoms.size(); ++j) {
        if ((transformed_pos.col(i) - atoms.frac_pos.col(j)).norm() <
            tolerance) {
          HKL offset{offsets(0, i), offsets(1, i), offsets(2, i)};
          table.m_graph.add_edge(i, j,
                                 SiteMappingEdge{i, j, symop.to_int(), offset});
          break;
        }
      }
    }
  }

  return table;
}

SiteMappingTable
SiteMappingTable::build_molecule_table(const Crystal &crystal) {
  SiteMappingTable table;
  const auto &molecules = crystal.unit_cell_molecules();
  const auto &symops = crystal.symmetry_operations();

  // Add vertices
  for (size_t i = 0; i < molecules.size(); ++i) {
    table.m_graph.add_vertex({i});
  }

  const double tolerance = 1e-6; // Tolerance for position comparison

  for (size_t i = 0; i < molecules.size(); ++i) {
    const auto &mol_i = molecules[i];
    Vec3 frac_centroid_i = crystal.to_fractional(mol_i.centroid());

    for (size_t j = 0; j < symops.size(); ++j) {
      const auto &symop = symops[j];
      Vec3 transformed_centroid = symop.apply(frac_centroid_i);

      // Compute offset and wrap position back to unit cell
      IVec3 offset = transformed_centroid.array().floor().cast<int>();
      transformed_centroid -= offset.cast<double>();

      for (size_t k = 0; k < molecules.size(); ++k) {
        const auto &mol_k = molecules[k];
        if (!mol_i.is_equivalent_to(mol_k))
          continue;

        Vec3 frac_centroid_k = crystal.to_fractional(mol_k.centroid());

        // Check if transformed centroid matches mol_k's centroid
        if ((transformed_centroid - frac_centroid_k).norm() <= tolerance) {
          // Add edge to the graph
          HKL hkl_offset{offset(0), offset(1), offset(2)};
          table.m_graph.add_edge(
              i, k, SiteMappingEdge{i, k, symop.to_int(), hkl_offset});

          // We found a match, no need to check other molecules
          break;
        }
      }
    }
  }

  return table;
}

std::vector<SiteMappingEdge> SiteMappingTable::get_edges(size_t source,
                                                         size_t target) const {
  std::vector<SiteMappingEdge> edges;
  for (const auto &[edge_descriptor, edge] : m_graph.edges()) {
    if (edge.source == source && edge.target == target) {
      edges.push_back(edge);
    }
  }
  return edges;
}

std::optional<size_t> SiteMappingTable::get_target(size_t source, int symop,
                                                   const HKL &offset) const {
  for (const auto &[edge_descriptor, edge] : m_graph.edges()) {
    if (edge.source == source && edge.symop == symop && edge.offset == offset) {
      return edge.target;
    }
  }
  return std::nullopt;
}

std::optional<std::pair<size_t, HKL>>
SiteMappingTable::get_target_and_offset(size_t source, int symop) const {
  for (const auto &[edge_descriptor, edge] : m_graph.edges()) {
    if (edge.source == source && edge.symop == symop) {
      return std::make_pair(edge.target, edge.offset);
    }
  }
  return std::nullopt;
}

std::vector<size_t> SiteMappingTable::get_neighbors(size_t source) const {
  std::vector<size_t> neighbors;
  for (const auto &[edge_descriptor, edge] : m_graph.edges()) {
    if (edge.source == source) {
      neighbors.push_back(edge.target);
    }
  }
  return neighbors;
}

std::vector<std::pair<int, HKL>>
SiteMappingTable::get_symmetry_operations(size_t source) const {
  std::vector<std::pair<int, HKL>> symops;
  for (const auto &[edge_descriptor, edge] : m_graph.edges()) {
    if (edge.source == source) {
      symops.emplace_back(edge.symop, edge.offset);
    }
  }
  return symops;
}

size_t SiteMappingTable::size() const { return m_graph.vertices().size(); }

} // namespace occ::crystal
