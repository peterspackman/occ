#include <fmt/os.h>
#include <occ/cg/distance_partition.h>
#include <occ/cg/neighbor_atoms.h>

namespace occ::cg {

SolventSurfacePartitioner::SolventSurfacePartitioner(
    const crystal::Crystal &crystal,
    const SolventSurfacePartitioner::NeighborList &neighbors)
    : m_crystal(crystal), m_neighbors(neighbors) {}

void SolventSurfacePartitioner::set_basename(const std::string &name) {
  m_basename = name;
}

void SolventSurfacePartitioner::set_should_write_surface_files(bool should) {
  m_should_write_surface_files = should;
}

void SolventSurfacePartitioner::set_use_normalized_distance(bool should) {
  m_use_dnorm = should;
}

std::vector<SolvationContribution>
partition(const SMDSolventSurfaces &surface) {
  return {};
}

void exchange_matching_forward_reverse_pairs(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors,
    std::vector<SolvationContribution> &energy_contribution) {
  for (int i = 0; i < neighbors.size(); i++) {
    auto &ci = energy_contribution[i];
    if (ci.has_been_exchanged())
      continue;
    for (int j = i; j < neighbors.size(); j++) {
      auto &cj = energy_contribution[j];
      if (cj.has_been_exchanged())
        continue;
      const auto &d1 = neighbors[i].dimer;
      const auto &d2 = neighbors[j].dimer;
      if (d1.equivalent_in_opposite_frame(d2)) {
        energy_contribution[i].exchange_with(energy_contribution[j]);
        break;
      }
    }
  }
}

struct PartitionedSurface {
  Mat3N positions;
  Vec energies;
  IVec molecule_index;
};

inline void write_surface_file(const std::string &filename,
                               const PartitionedSurface &surf) {
  using occ::units::angstroms;
  auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                               fmt::file::CREATE);
  const size_t N = surf.positions.cols();
  output.print("{}\nx y z e neighbor\n", N);

  for (size_t i = 0; i < N; i++) {
    occ::Vec3 x = surf.positions.col(i);
    int idx = surf.molecule_index(i);
    double e = surf.energies(i);

    output.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n", angstroms(x(0)),
                 angstroms(x(1)), angstroms(x(2)), e, idx);
  }
}

std::vector<SolvationContribution>
SolventSurfacePartitioner::partition_nearest_atom(
    const SolventSurfacePartitioner::NeighborList &nearest,
    const SMDSolventSurfaces &surface) {
  using occ::units::angstroms;
  std::vector<SolvationContribution> energy_contribution(m_neighbors.size());
  for (auto &contrib : energy_contribution) {
    contrib.set_antisymmetrize(should_antisymmetrize());
  }

  auto natoms = NeighborAtoms(nearest);

  auto closest_idx = [&](const Vec3 &x, const Mat3N &pos, const Vec &vdw) {
    Eigen::Index idx = 0;
    Vec norms = (natoms.positions.colwise() - x).colwise().norm();
    if (m_use_dnorm) {
      norms.array() -= vdw.array();
      norms.array() /= vdw.array();
    }
    norms.minCoeff(&idx);
    return idx;
  };

  // Process coulomb contributions
  PartitionedSurface coul;
  coul.positions = surface.coulomb.positions;
  coul.molecule_index = IVec(surface.coulomb.size());
  coul.energies = surface.coulomb.energies;

  for (size_t i = 0; i < surface.coulomb.size(); i++) {
    occ::Vec3 x = surface.coulomb.positions.col(i);
    Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
    auto m_idx = natoms.molecule_index(idx);

    energy_contribution[m_idx].add_coulomb(surface.coulomb.energies(i) +
                                           surface.electronic_energies(i));
    energy_contribution[m_idx].add_coulomb_area(surface.coulomb.areas(i));

    coul.molecule_index(i) = m_idx;
  }

  if (m_should_write_surface_files) {
    write_surface_file(fmt::format("{}_coulomb.txt", m_basename), coul);
  }

  // Process coulomb contributions
  PartitionedSurface cds;
  cds.positions = surface.cds.positions;
  cds.molecule_index = IVec(surface.cds.size());
  cds.energies = surface.cds.energies;

  for (size_t i = 0; i < surface.cds.size(); i++) {
    occ::Vec3 x = surface.cds.positions.col(i);
    Eigen::Index idx = closest_idx(x, natoms.positions, natoms.vdw_radii);
    auto m_idx = natoms.molecule_index(idx);
    energy_contribution[m_idx].add_cds(surface.cds.energies(i));
    energy_contribution[m_idx].add_cds_area(surface.cds.areas(i));

    cds.molecule_index(i) = m_idx;
  }

  if (m_should_write_surface_files) {
    write_surface_file(fmt::format("{}_cds.txt", m_basename), cds);
  }

  exchange_matching_forward_reverse_pairs(m_neighbors, energy_contribution);
  return energy_contribution;
}

std::vector<SolvationContribution> SolventSurfacePartitioner::partition(
    const SolventSurfacePartitioner::NeighborList &nearest,
    const SMDSolventSurfaces &surface) {
  return partition_nearest_atom(nearest, surface);
}

} // namespace occ::cg
