#pragma once
#include <istream>
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
  /// Optional user/growth morphology: (hkl -> support distance). When non-empty
  /// it replaces the equilibrium (Wulff) shape. Each face still sits at its
  /// lowest-energy termination, so every face needs a computed surface energy.
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

/// Read a particle shape for `MorphologyOptions::user_shifts`: one face per
/// line as `h k l distance`, one face per form (its symmetry-equivalent faces
/// are added), distances in any one unit since only their ratios matter.
/// Blank lines and anything after '#' are ignored; a malformed line, or a
/// distance that is not positive, is an error.
std::vector<std::pair<occ::crystal::HKL, double>>
read_morphology_shape(std::istream &input);

/// One named shape from a multi-shape file.
struct NamedShape {
  std::string name; ///< empty for a file that names no shapes
  std::vector<std::pair<occ::crystal::HKL, double>> shifts;
};

/// Read one or more shapes. A line `shape <name>` opens a named shape and the
/// faces that follow belong to it; a file with no such line is a single
/// unnamed shape, so every file `read_morphology_shape` accepts is still
/// valid here.
///
/// Scanning habits is what a user shape is for, and the shape itself is the
/// cheap part of the calculation: the pair energies, the monomers and the
/// surface enumeration are all independent of it. Reading the whole set at
/// once lets those be done once rather than once per habit.
std::vector<NamedShape> read_morphology_shapes(std::istream &input);

} // namespace occ::driver
