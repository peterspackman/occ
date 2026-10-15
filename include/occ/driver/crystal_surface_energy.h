#pragma once
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cmath>
#include <nlohmann/json.hpp>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <string>

namespace occ::driver {

struct FacetEnergies {
  occ::crystal::HKL hkl;
  double offset{0.0};
  std::vector<std::vector<size_t>> interaction_energy_counts;
  double energy{0.0};
  double area{0.0};

  /// Solvation descriptors summed over the contacts this cut breaks, in the
  /// units the solvation model documents. Empty when no solvation model
  /// carried any.
  ankerl::unordered_dense::map<std::string, double> descriptors{};
};

struct CrystalSurfaceEnergies {
  occ::crystal::Crystal crystal;
  std::vector<double> unique_interaction_energies;
  std::vector<FacetEnergies> facets;
};

/// Whether a cut (energy, offset) should replace the best one so far.
///
/// Symmetry-equivalent terminations of a face have equal energies up to
/// summation-order noise, which differs run to run under threading. Treat
/// energies within a relative 1e-9 as tied and keep the lower offset, so the
/// chosen termination, and the particle built from it, is reproducible.
inline bool is_lower_cut(double energy, double offset, double best_energy,
                         double best_offset) {
  const double tol =
      1e-9 * std::max(std::abs(energy), std::abs(best_energy));
  if (std::abs(energy - best_energy) <= tol)
    return offset < best_offset;
  return energy < best_energy;
}

/// Surface energies for the morphologically important faces.
///
/// Faces are considered in Bravais-Friedel-Donnay-Harker order, largest
/// interplanar spacing first. `min_interplanar_spacing` (Angstrom, positive
/// to use) is the crystallographically meaningful cut: every face with
/// d >= the threshold is included, so the selection never depends on where a
/// count happens to land.
///
/// `max_number_of_surfaces` is the older count-based cut, kept for
/// compatibility. It can split a Friedel pair — two distinct forms that are
/// exactly degenerate in d whenever the point group is non-centrosymmetric —
/// and produce an asymmetric Wulff construction, so it warns when it does.
/// Set `min_interplanar_spacing` instead where you can.
CrystalSurfaceEnergies calculate_crystal_surface_energies(
    const std::string &filename, const occ::crystal::Crystal &crystal,
    const occ::crystal::CrystalDimers &uc_dimers, int max_number_of_surfaces,
    int sign = -1, double min_interplanar_spacing = 0.0);

void to_json(nlohmann::json &j, const FacetEnergies &);
void to_json(nlohmann::json &j, const CrystalSurfaceEnergies &);

} // namespace occ::driver
