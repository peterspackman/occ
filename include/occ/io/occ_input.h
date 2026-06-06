#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <occ/core/spinorbital.h>
#include <occ/crystal/crystal.h>
#include <occ/numint/grid_settings.h>

namespace occ::io {
using occ::core::Element;
using occ::core::SpinorbitalKind;

using Position = std::array<double, 3>;
using PointChargeList = std::vector<occ::core::PointCharge>;

struct ElectronInput {
  int multiplicity{1};
  double charge{0.0};
  SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
};

struct GeometryInput {
  std::vector<Position> positions;
  std::vector<Element> elements;
  occ::core::Molecule molecule() const;
  void set_molecule(const occ::core::Molecule &);
  PointChargeList point_charges;
  std::string point_charge_filename{""};
};

struct OutputInput {
  std::vector<std::string> formats{"json"};
};

struct DriverInput {
  std::string driver{"energy"};
};

struct COSXSettings {
  double screen_threshold{1e-4};      // Shell extent screening threshold (looser = smaller extents = more screening)
  double margin{1.0};                 // Geometric margin (Bohr)
  double f_threshold{1e-10};          // F-intermediate threshold
};

// Acceleration policy for SCF two-electron integrals (density fitting / COSX).
//   Auto - choose automatically (ORCA-style): density-fit the Coulomb term, and
//          for exact exchange (HF / hybrid DFT) use DF-K below a basis-function
//          crossover and seminumerical COSX above it.
//   None - conventional 4-center integrals (no DF, no COSX).
//   JK   - force density fitting for both J and K.
//   COSX - force DF for J and seminumerical COSX for K (exact exchange only).
enum class RIPolicy { Auto, None, JK, COSX };

struct MethodInput {
  std::string name{"rhf"};
  occ::numint::GridSettings dft_grid;
  double integral_precision{1e-12};
  int scf_maxiter{100}; // Maximum number of SCF iterations
  // DFT XC integration: per-grid-batch shell screening tolerance (the |phi|
  // decay cutoff used to drop negligible basis functions over a spatial
  // batch). Larger = more aggressive screening / faster, less accurate. <=0
  // disables screening (dense XC build).
  double dft_xc_screening_threshold{1e-10};
  double orbital_smearing_sigma{0.0};
  RIPolicy ri_policy{RIPolicy::Auto}; // Automatic DF/COSX selection (see RIPolicy)
  bool use_direct_df_kernels{false}; // Use direct DF kernels instead of stored for testing
  bool use_split_ri_j{false}; // Use Split-RI-J for Coulomb matrix (Neese 2003)
  bool use_cosx{false}; // Use COSX seminumerical exchange
  occ::numint::COSXGridLevel cosx_grid_level{occ::numint::COSXGridLevel::Grid1}; // COSX grid quality
  COSXSettings cosx; // COSX settings
  double mp2_max_memory_gb{1.0}; // MP2 B-tensor / half-transform memory budget
  std::string mp2_spin_scaling{"none"}; // MP2 spin scaling: none | scs | sos
  // MP2 integral backend: "auto" keeps the current behaviour (RI-MP2 when an
  // ri_basis is given, else conventional); "thc" uses LS-THC-MP2 (Laplace
  // denominator + THC factors), reusing the THC options below.
  std::string mp2_backend{"auto"};
  double mp2_thc_c_isdf{6.0};               // THC rank = c * nbf
  std::string mp2_thc_method{"cholesky"};   // THC ISDF selector: cholesky | qr
  int mp2_laplace_points{14};               // Laplace quadrature points
  std::string ccsd_backend{"exact"};    // CCSD integral backend: exact | df | thc
  double ccsd_max_memory_gb{1.0};       // CCSD integral-build memory budget
  int ccsd_frozen_core{-1};             // -1 auto (chemical core), 0 none, N freeze N
  // THC rank = c * nbf. c~6 gives sub-mHa CCSD(T) vs DF across systems and is
  // the accuracy/cost sweet spot; the error floors there (higher c is slower,
  // ~c^2, and no more accurate -- the LS-THC metric is ill-conditioned).
  double ccsd_thc_c_isdf{6.0};
  std::string ccsd_thc_method{"cholesky"}; // THC ISDF selector: cholesky | qr
  int ccsd_thc_grid_angular{110};          // THC candidate-grid max angular pts
  double ccsd_thc_grid_radial{1e-7};       // THC candidate-grid radial precision
};

struct BasisSetInput {
  std::string name{"3-21G"};
  std::string df_name{""};
  std::string ri_basis{""};
  std::string basis_set_directory{""};
  double df_auto_threshold{1e-4};  ///< Cholesky threshold for auto aux basis
  bool spherical{false};
};

struct SolventInput {
  std::string solvent_name{""};
  std::string output_surface_filename{""};
  bool radii_scaling{false};
};

struct RuntimeInput {
  int threads{1};
  std::string output_filename{""};
};

struct OptimizationInput {
  // Convergence criteria 
  double gradient_max{0.15e-3};     // Maximum gradient component (Ha/Angstrom) - more strict, ~ORCA's 3e-4 Ha/bohr
  double gradient_rms{0.05e-3};     // RMS gradient (Ha/Angstrom) - more strict, ~ORCA's 1e-4 Ha/bohr
  double step_max{1.8e-3};          // Maximum step (Angstrom)
  double step_rms{1.2e-3};          // RMS step (Angstrom)
  double energy_change{1e-6};       // Energy convergence (Hartree)
  bool use_energy_criterion{false}; // Whether to use energy criterion
  int max_iterations{100};          // Maximum optimization steps
  
  // Integral accuracy control
  double gradient_integral_precision{1e-10}; // Final gradient integral cutoff
  double early_gradient_integral_precision{1e-8}; // Looser cutoff for early steps
  double tight_gradient_threshold{1e-3}; // Switch to tight precision when |ΔE| < this (Hartree)
  
  // Output options
  bool write_wavefunction_steps{false}; // Write wavefunction at each optimization step
  
  // Frequency analysis
  bool compute_frequencies{false};  // Compute vibrational frequencies after optimization
};

struct DispersionCorrectionInput {
  bool evaluate_correction{false};
  double xdm_a1{1.0};
  double xdm_a2{1.0};
};

struct CrystalInput {
  occ::crystal::AsymmetricUnit asymmetric_unit;
  occ::crystal::SpaceGroup space_group;
  occ::crystal::UnitCell unit_cell;
};

struct PairInput {
  std::string source_a{"none"};
  Mat3 rotation_a{Mat3::Identity()};
  Vec3 translation_a{Vec3::Zero()};
  int ecp_electrons_a{0};
  std::string source_b{"none"};
  Mat3 rotation_b{Mat3::Identity()};
  Vec3 translation_b{Vec3::Zero()};
  int ecp_electrons_b{0};
  std::string model_name{"ce-b3lyp"};
};

struct IsosurfaceInput {
  GeometryInput interior_geometry;
  GeometryInput exterior_geometry;
};

struct OccInput {
  std::string verbosity{"normal"};
  DriverInput driver;
  RuntimeInput runtime;
  ElectronInput electronic;
  GeometryInput geometry;
  PairInput pair;
  MethodInput method;
  BasisSetInput basis;
  SolventInput solvent;
  DispersionCorrectionInput dispersion;
  CrystalInput crystal;
  IsosurfaceInput isosurface;
  OutputInput output;
  OptimizationInput optimization;
  std::string name{""};
  std::string filename{""};
  std::string chelpg_filename{""};
};

template <typename T> OccInput build(const std::string &filename) {
  return T(filename).as_occ_input();
}

template <typename T> OccInput build(std::istream &file) {
  return T(file).as_occ_input();
}

} // namespace occ::io
