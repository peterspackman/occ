#pragma once
#include <occ/crystal/hkl.h>
#include <string>
#include <vector>

namespace occ::cg {

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
  /// Line tension (kJ/mol per Angstrom): the inclusion-exclusion correction to
  /// the flat-surface model, per unit edge length (typically negative). This
  /// per-edge value is discretization-sensitive; prefer the length-weighted
  /// mean and the per-size e_edge totals.
  double lambda{0.0};
};

/// A symmetry-unique corner type of the particle shape.
struct CornerMorphology {
  std::vector<occ::crystal::HKL> hkls; ///< facets meeting at the corner
  int count{0};                        ///< number of such corners on the shape
  double epsilon{0.0};                 ///< corner energy (kJ/mol), per corner
};

/// Broken-bond excess energy of a finite particle at one size, decomposed by
/// inclusion-exclusion so e_excess == e_surface + e_edge + e_corner exactly.
struct ParticleSample {
  double size_scale{0.0};
  int n_molecules{0};
  double e_excess{0.0};  ///< kJ/mol (== e_surface + e_edge + e_corner)
  double e_surface{0.0}; ///< flat-surface term
  double e_edge{0.0};    ///< line-tension (edge) correction
  double e_corner{0.0};  ///< corner correction
  double e_surface_analytic{0.0}; ///< exact sum_f gamma_f*A_f (optimal-cut surface energy)
  double area{0.0};               ///< Angstrom^2
  double edge_length{0.0};        ///< Angstrom
  int n_corners{0};
};

/// Particle size/shape-dependent (surface + edge + corner) energies.
struct MorphologyResult {
  std::string shape{"wulff"};
  double mu_bulk{0.0};          ///< per-molecule lattice energy (0.5 * crystal_energy), kJ/mol
  double molecular_volume{0.0}; ///< Angstrom^3
  std::vector<FacetMorphology> facets;
  std::vector<EdgeMorphology> edges;
  std::vector<CornerMorphology> corners;
  std::vector<ParticleSample> samples;

  bool empty() const { return facets.empty() && samples.empty(); }
};

} // namespace occ::cg
