#pragma once
#include <occ/cg/solvation_contribution.h>
#include <occ/cg/solvation_data.h>
#include <occ/crystal/crystal.h>

namespace occ::cg {

class SolventSurfacePartitioner {
public:
  using NeighborList = crystal::CrystalDimers::MoleculeNeighbors;
  explicit SolventSurfacePartitioner(const NeighborList &full_neighbors);

  void set_basename(const std::string &);

  void set_should_write_surface_files(bool);
  [[nodiscard]] inline bool should_write_surface_files() const {
    return m_should_write_surface_files;
  }

  void set_use_normalized_distance(bool);
  [[nodiscard]] inline bool use_normalized_distance() const {
    return m_use_dnorm;
  }

  /// Assign every element of every cavity to its nearest neighbour molecule.
  /// Each cavity contributes its energy and descriptor channels verbatim,
  /// plus an `<cavity>_area` descriptor, so a solvation model carrying new
  /// channels needs no change here.
  [[nodiscard]] std::vector<SolvationContribution>
  partition(const NeighborList &nearest, const SolvationData &surface);

  [[nodiscard]] inline bool should_antisymmetrize() const {
    return m_antisymmetrize;
  }
  inline void set_should_antisymmetrize(bool should) {
    m_antisymmetrize = should;
  }

private:
  std::vector<SolvationContribution>
  partition_nearest_atom(const NeighborList &nearest,
                         const SolvationData &surface);

  std::string m_basename{"molecule_solvent"};
  bool m_antisymmetrize{true};
  bool m_use_dnorm{true};
  bool m_should_write_surface_files{true};
  const crystal::CrystalDimers::MoleculeNeighbors &m_neighbors;
};

void exchange_matching_forward_reverse_pairs(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors,
    std::vector<SolvationContribution> &energy_contribution);

} // namespace occ::cg
