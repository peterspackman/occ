#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/xtb/solvation_interface.h>

namespace occ::cg {

struct SolventSurface {
  occ::Mat3N positions;
  occ::Vec energies;
  occ::Vec areas;

  [[nodiscard]] double total_energy() const;
  [[nodiscard]] double total_area() const;
  [[nodiscard]] size_t size() const;
};

struct SMDSolventSurfaces {
  SolventSurface coulomb;
  SolventSurface cds;
  occ::Vec electronic_energies;

  // Energy components
  double total_solvation_energy{0.0};
  double electronic_contribution{0.0};
  double gas_phase_contribution{0.0};
  double free_energy_correction{0.0};

  [[nodiscard]] double total_energy() const;
};

/// Adapter from the xtb backend's per-element surface bundle to the cg
/// shape consumed by `SolventSurfacePartitioner` and the JSON writers.
///
/// Behaviour:
///   • `xtb::SolvationSurface.energies` already carries the full per-element
///     ES contribution (½ σ_i · φ_i, atom-resolved Mulliken-charge source),
///     so it flows straight into `coulomb.energies` and `electronic_energies`
///     stays zero — the cg partitioner sums them anyway, see
///     `distance_partition.cpp:108`.
///   • CPCM-X has no CDS branch (`xtb_surfaces.cds == std::nullopt`); the
///     returned `cds` surface comes out empty.
///   • `total_solvation_energy` is set to `xtb_surfaces.total_energy()`; the
///     remaining DFT-flavoured fields (`electronic_contribution`,
///     `gas_phase_contribution`, `free_energy_correction`) stay at their
///     default zero since xtb doesn't produce a corresponding decomposition.
[[nodiscard]] SMDSolventSurfaces
from_xtb_surfaces(const occ::xtb::SolvationSurfaces &xtb_surfaces);

} // namespace occ::cg
