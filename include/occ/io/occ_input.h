#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/core/molecule.h>
#include <occ/core/point_charge.h>
#include <occ/crystal/crystal.h>
#include <occ/io/grid_settings.h>
#include <occ/qm/spinorbital.h>
#include <vector>

namespace occ::io {
using occ::core::Element;
using occ::qm::SpinorbitalKind;

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

struct MethodInput {
  std::string name{"rhf"};
  GridSettings dft_grid;
  double integral_precision{1e-12};
  double orbital_smearing_sigma{0.0};
  bool use_direct_df_kernels{false}; // Use direct DF kernels instead of stored for testing
};

struct BasisSetInput {
  std::string name{"3-21G"};
  std::string df_name{""};
  std::string ri_basis{""};
  std::string basis_set_directory{""};
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
