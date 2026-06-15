#pragma once
#include <nlohmann/json.hpp>
#include <occ/cg/result_types.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <occ/driver/crystal_surface_energy.h>
#include <string>
#include <utility>
#include <vector>

namespace occ::driver {

/// \brief Options for the particle size/shape-dependent energy calculation.
struct MorphologyOptions {
  std::vector<int> sizes{1000, 2000, 4000, 8000, 16000, 32000};
  double sign{1.0};   ///< +1 for solvated facet energies, -1 for vacuum
  /// Optional user/growth morphology: (hkl -> support distance). When non-empty it
  /// replaces the equilibrium (Wulff) shape.
  std::vector<std::pair<occ::crystal::HKL, double>> user_shifts{};
};

/// A facet of the (Wulff or user) particle shape.
struct FacetMorphology {
  occ::crystal::HKL hkl;
  double gamma{0.0}; ///< surface energy / support distance (J/m^2)
  double area{0.0};  ///< face area of the unit-scale shape
};

/// A symmetry-unique edge type of the particle shape.
struct EdgeMorphology {
  occ::crystal::HKL hkl_a, hkl_b;
  double length{0.0}; ///< total unit-scale length of this edge type
  double lambda{0.0}; ///< line tension (kJ/mol per Angstrom)
};

/// A symmetry-unique corner type of the particle shape.
struct CornerMorphology {
  std::vector<occ::crystal::HKL> hkls; ///< facets meeting at the corner
  int count{0};                        ///< number of such corners on the shape
  double epsilon{0.0};                 ///< corner energy (kJ/mol)
};

/// Broken-bond excess energy of a finite particle at one size.
struct ParticleSample {
  double size_scale{0.0};
  int n_molecules{0};
  double e_excess{0.0};          ///< kJ/mol (== e_surface + e_edge + e_corner)
  double e_surface{0.0};         ///< broken-bond surface attribution
  double e_edge{0.0};
  double e_corner{0.0};
  double e_surface_analytic{0.0}; ///< exact sum_f gamma_f*A_f (optimal-cut surface energy)
  double area{0.0};               ///< Angstrom^2
  double edge_length{0.0};        ///< Angstrom
  int n_corners{0};
};

struct MorphologyResult {
  std::string shape{"wulff"};
  double mu_bulk{0.0};          ///< per-molecule lattice energy (0.5 * crystal_energy), kJ/mol
  double molecular_volume{0.0}; ///< Angstrom^3
  std::vector<FacetMorphology> facets;
  std::vector<EdgeMorphology> edges;
  std::vector<CornerMorphology> corners;
  std::vector<ParticleSample> samples;
};

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

void to_json(nlohmann::json &, const MorphologyResult &);

} // namespace occ::driver
