#pragma once
#include <nlohmann/json.hpp>
#include <occ/cg/morphology_types.h>
#include <occ/cg/result_types.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <occ/driver/crystal_surface_energy.h>
#include <string>
#include <utility>
#include <vector>

namespace occ::cg {
void to_json(nlohmann::json &, const MorphologyResult &);
} // namespace occ::cg

namespace occ::driver {

/// \brief Options for the particle size/shape-dependent energy calculation.
struct MorphologyOptions {
  std::vector<int> sizes{1000, 2000, 4000, 8000, 16000, 32000};
  double sign{1.0}; ///< +1 for solvated facet energies, -1 for vacuum
  /// Optional user/growth morphology: (hkl -> support distance). When non-empty it
  /// replaces the equilibrium (Wulff) shape.
  std::vector<std::pair<occ::crystal::HKL, double>> user_shifts{};
};

using occ::cg::CornerMorphology;
using occ::cg::EdgeMorphology;
using occ::cg::FacetMorphology;
using occ::cg::MorphologyResult;
using occ::cg::ParticleSample;

/// Compute the particle size/shape-dependent (surface + edge + corner) energy.
///
/// \param uc_dimers must already carry interaction energies (as produced by the cg flow
///                  via InteractionMapper); `dimer.interaction_energy("Total")` is read.
MorphologyResult compute_crystal_morphology(
    const occ::crystal::Crystal &crystal,
    const occ::crystal::CrystalDimers &uc_dimers,
    const CrystalSurfaceEnergies &surface_energies,
    const occ::cg::CrystalGrowthResult &growth_result,
    const MorphologyOptions &options = {});

} // namespace occ::driver
