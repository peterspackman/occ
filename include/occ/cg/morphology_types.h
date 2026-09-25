#pragma once
#include <array>
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

/// One stamped neighbour interaction, as the decomposition sees it. Emitted
/// only on request: it is the input an external check of the broken-bond sum
/// needs, and the per-molecule pair list written elsewhere is a different
/// object that need not agree bond by bond.
struct NeighbourBond {
  int source{0};         ///< unit-cell molecule the bond belongs to
  int target{0};         ///< unit-cell molecule at the other end
  int shift[3]{0, 0, 0}; ///< cell offset of the target
  double energy{0.0};    ///< interaction_energy("Total"), kJ/mol
};

/// One active face of the particle shape, as the decomposition uses it: the
/// support distance is per unit scale, and the optimal-cut offset and
/// interplanar spacing are what snap it to a molecular termination.
struct ShapeFace {
  occ::crystal::HKL hkl;
  double normal[3]{0.0, 0.0, 0.0}; ///< unit normal, Cartesian
  double distance{0.0};            ///< support distance at unit scale
  double offset{0.0};              ///< optimal cut offset, fraction of d
  double d_spacing{0.0};           ///< interplanar spacing (Angstrom)
};

/// Particle size/shape-dependent (surface + edge + corner) energies.
struct MorphologyResult {
  std::string shape{"wulff"};
  std::string name;             ///< habit name, when a multi-shape file gave one
  double mu_bulk{0.0};          ///< per-molecule lattice energy (0.5 * crystal_energy), kJ/mol
  double molecular_volume{0.0}; ///< Angstrom^3
  std::vector<FacetMorphology> facets;
  std::vector<EdgeMorphology> edges;
  std::vector<CornerMorphology> corners;
  std::vector<ParticleSample> samples;
  std::vector<NeighbourBond> bonds; ///< only when MorphologyOptions::emit_bonds
  /// Fractional centroids of the unit-cell molecules, in the order the bonds
  /// index them. Emitted with the bonds: a molecule reassembled independently
  /// can land in a different periodic image, which changes the cluster.
  std::vector<std::array<double, 3>> uc_centroids;
  std::vector<ShapeFace> shape_faces; ///< emitted with the bonds

  bool empty() const { return facets.empty() && samples.empty(); }
};

} // namespace occ::cg
