#pragma once
#include <occ/cg/solvent_surface.h>
#include <occ/core/linear_algebra.h>
#include <string>
#include <string_view>
#include <vector>

namespace occ::cg {

/// A named per-element quantity defined on one cavity.
struct SurfaceField {
  std::string name;
  Vec values;
};

/// One cavity's worth of per-element solvation data.
///
/// A solvation model may use more than one cavity — SMD builds separate
/// electrostatic and CDS cavities with different radii, so their elements do
/// not correspond — which is why geometry lives here rather than being shared.
struct CavitySurface {
  std::string name; ///< used for the per-cavity area channel and dump files
  Mat3N positions;  ///< Bohr
  Vec areas;        ///< Bohr²
  std::vector<SurfaceField> energies;    ///< Hartree, summed into the total
  std::vector<SurfaceField> descriptors; ///< carried, never summed

  [[nodiscard]] size_t size() const { return positions.cols(); }
  [[nodiscard]] double total_area() const { return areas.sum(); }
  [[nodiscard]] double total_energy() const;
};

/// Every cavity a solvation model produced for one molecule, plus the scalar
/// bookkeeping cg reports.
///
/// This is the shape the partitioner consumes. It is deliberately not the
/// on-disk format: `SMDSolventSurfaces` remains the cached representation so
/// existing caches keep loading.
struct SolvationData {
  std::vector<CavitySurface> cavities;

  double total_solvation_energy{0.0};
  double electronic_contribution{0.0};
  double gas_phase_contribution{0.0};
  double free_energy_correction{0.0};

  [[nodiscard]] double total_energy() const;
  [[nodiscard]] const CavitySurface *find(std::string_view name) const;
};

/// SMD/CPCM-X cache format to the partitionable shape.
///
/// The electrostatic cavity carries `coulomb.energies + electronic_energies`
/// as a single `coulomb` channel, matching what the partitioner has
/// always summed. Cavity names are kept as `coulomb` and `cds` so the dump
/// file names do not change.
[[nodiscard]] SolvationData to_solvation_data(const SMDSolventSurfaces &);

} // namespace occ::cg
