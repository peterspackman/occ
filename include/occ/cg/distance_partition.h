#pragma once
#include <occ/cg/solvation_contribution.h>
#include <occ/cg/solvent_surface.h>
#include <occ/crystal/crystal.h>

namespace occ::cg {

class SolventSurfacePartitioner {
public:
  using NeighborList = crystal::CrystalDimers::MoleculeNeighbors;
  SolventSurfacePartitioner(const crystal::Crystal &crystal,
                            const NeighborList &full_neighbors);

  void set_basename(const std::string &);
  [[nodiscard]] inline const auto &basename() const { return m_basename; }

  void set_should_write_surface_files(bool);
  [[nodiscard]] inline bool should_write_surface_files() const {
    return m_should_write_surface_files;
  }

  void set_use_normalized_distance(bool);
  [[nodiscard]] inline bool use_normalized_distance() const {
    return m_use_dnorm;
  }

  [[nodiscard]] std::vector<SolvationContribution>
  partition(const NeighborList &nearest, const SMDSolventSurfaces &surface);

  [[nodiscard]] inline bool should_antisymmetrize() const {
    return m_antisymmetrize;
  }
  inline void set_should_antisymmetrize(bool should) {
    m_antisymmetrize = should;
  }

private:
  std::vector<SolvationContribution>
  partition_nearest_atom(const NeighborList &nearest,
                         const SMDSolventSurfaces &surface);

  std::string m_basename{"molecule_solvent"};
  bool m_antisymmetrize{true};
  bool m_use_dnorm{true};
  bool m_should_write_surface_files{true};
  const crystal::Crystal &m_crystal;
  const crystal::CrystalDimers::MoleculeNeighbors &m_neighbors;
};

void exchange_matching_forward_reverse_pairs(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors,
    std::vector<SolvationContribution> &energy_contribution);

} // namespace occ::cg
