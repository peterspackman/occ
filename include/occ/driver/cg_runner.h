#pragma once
#include <occ/cg/result_types.h>
#include <occ/interaction/lattice_convergence_settings.h>
#include <string>

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
  DMAReferenceLevel dma_reference;
  std::string charge_string{""};
  std::string wavefunction_choice{"gas"};
  double cg_radius{3.8};
  int max_facets{0};
  bool compute_morphology{false};
  bool write_dump_files{false};
  bool spherical{false};
  bool write_kmcpp_file{false};
  bool use_xtb{false};
  bool dry_run{false};
  bool asymmetric_solvent_contribution{false};
  bool gamma_point_molecules{true};
  std::string xtb_solvation_model{"cpcmx"};
  bool list_solvents{false};
  bool crystal_is_atomic{false};
};

occ::cg::CrystalGrowthResult run_cg(CGConfig const &);

} // namespace occ::driver
