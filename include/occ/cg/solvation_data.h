#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/scrf/surfaces.h>
#include <occ/xtb/solvation_interface.h>
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

  [[nodiscard]] double total_energy() const;
  [[nodiscard]] const CavitySurface *find(std::string_view name) const;
};

/// Convenience accessors for the two SMD/CPCM-X cavities.
[[nodiscard]] CavitySurface *coulomb_cavity(SolvationData &);
[[nodiscard]] const CavitySurface *coulomb_cavity(const SolvationData &);
[[nodiscard]] CavitySurface *cds_cavity(SolvationData &);
[[nodiscard]] const CavitySurface *cds_cavity(const SolvationData &);

/// Add a named cavity with a single energy channel of the same name.
///
/// \warning Returns a reference into `cavities`, so it is invalidated by the
/// next `add_cavity` exactly as `std::vector::emplace_back` would be. Finish
/// with one cavity before adding the next.
CavitySurface &add_cavity(SolvationData &, const std::string &name,
                          const Mat3N &positions, const Vec &areas,
                          const Vec &energies);

/// Build from the unified `occ::scrf::SolvationSurfaces` bundle produced by
/// the HF/DFT and xTB pipelines. CPCM-X has no CDS branch, so that cavity is
/// simply absent.
[[nodiscard]] SolvationData
from_scrf_surfaces(const occ::scrf::SolvationSurfaces &);

[[nodiscard]] inline SolvationData
from_xtb_surfaces(const occ::xtb::SolvationSurfaces &surfaces) {
  return from_scrf_surfaces(surfaces);
}

} // namespace occ::cg
