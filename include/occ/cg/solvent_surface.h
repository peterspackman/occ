#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/scrf/surfaces.h>
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

/// Adapter from the unified `occ::scrf::SolvationSurfaces` per-element bundle
/// (produced by both the xTB and HF/DFT pipelines after Phase 3) to the cg
/// shape consumed by `SolventSurfacePartitioner` and the JSON writers.
///
/// Behaviour:
///   • `scrf::SolvationSurface.energies` carries the full per-element ES
///     contribution `½ σ_i · φ_total_i` — identical algebraically to the
///     legacy `nuc_i + elec_i + pol_i` per-element decomposition — so it
///     flows straight into `coulomb.energies` and `electronic_energies`
///     stays zero. The cg partitioner sums `coulomb.energies(i) +
///     electronic_energies(i)` per element anyway, see
///     `distance_partition.cpp:108`.
///   • CPCM-X has no CDS branch (`scrf_surfaces.cds == std::nullopt`); the
///     returned `cds` surface comes out empty.
///   • `total_solvation_energy` is set to `scrf_surfaces.total_energy()`; the
///     remaining DFT-flavoured fields (`electronic_contribution`,
///     `gas_phase_contribution`, `free_energy_correction`) stay at their
///     default zero — the DFT caller fills them in via
///     `SMDCalculator::calculate_free_energy_components`.
[[nodiscard]] SMDSolventSurfaces
from_scrf_surfaces(const occ::scrf::SolvationSurfaces &scrf_surfaces);

/// Backwards-compatible alias. `occ::xtb::SolvationSurfaces` is a type alias
/// for `occ::scrf::SolvationSurfaces` (Phase 2), so this just forwards.
[[nodiscard]] inline SMDSolventSurfaces
from_xtb_surfaces(const occ::xtb::SolvationSurfaces &xtb_surfaces) {
  return from_scrf_surfaces(xtb_surfaces);
}

} // namespace occ::cg
