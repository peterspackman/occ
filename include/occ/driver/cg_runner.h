#pragma once
#include <occ/cg/result_types.h>
#include <occ/interaction/lattice_convergence_settings.h>
#include <string>
#include <vector>

namespace occ::driver {

/// Reference QM level for the monomer multipoles in the DMA+exp-6 model.
/// \c model implies \c method and \c basis; non-empty overrides win.
struct DMAReferenceLevel {
  std::string model{"ce-b3lyp"};
  std::string method{""}; ///< empty -> taken from model
  std::string basis{""};  ///< empty -> taken from model
};

struct CGConfig {
  interaction::LatticeConvergenceSettings lattice_settings;
  std::string solvent{"water"};
  std::string solvation_model{"smd"};
  bool print_solvation_descriptors{false};
  double temperature{298.15};
  double solvent_probe_radius{0.0};
  DMAReferenceLevel dma_reference;
  std::string charge_string{""};
  std::string wavefunction_choice{"gas"};
  double cg_radius{3.8};
  /// Count-based surface selection. Deprecated: it can split a Friedel pair
  /// and skew the Wulff construction. Prefer `min_interplanar_spacing`.
  int max_facets{0};
  /// Include every face with d >= this (Angstrom). Takes precedence over
  /// `max_facets`, and is the crystallographically meaningful cut.
  double min_interplanar_spacing{0.0};
  bool compute_morphology{false};
  /// Particle sizes (molecules) sampled by the morphology; empty keeps the
  /// default series.
  std::vector<int> morphology_sizes{};
  /// File of `h k l distance` faces to use instead of the Wulff shape.
  std::string morphology_shape{};
  bool morphology_emit_bonds{false};
  bool write_dump_files{false};
  /// Keep monomer wavefunctions, monomer energies, solvation surfaces and
  /// pair energies in memory only, rather than as reusable files.
  bool no_cache{false};
  bool spherical{false};
  bool write_kmcpp_file{false};
  bool use_xtb{false};
  bool dry_run{false};
  bool asymmetric_solvent_contribution{false};
  bool gamma_point_molecules{true};
  bool list_solvents{false};
  bool crystal_is_atomic{false};
};

occ::cg::CrystalGrowthResult run_cg(CGConfig const &);

} // namespace occ::driver
