#pragma once
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/cosmors.h>
#include <occ/solvent/cosmors_segments.h>
#include <string>
#include <vector>

namespace occ::driver {

/// Settings for the ideal-conductor COSMO calculation the segment
/// descriptors are built from.
struct ConductorSettings {
  std::string method{"b3lyp"};
  std::string basis{"def2-svp"};
  bool pure_spherical{true};
  double probe_radius_angs{0.0};
  int angular_points{590};
  /// Constrain the surface charge to -q (Gauss's law for a conductor).
  bool constrain_charge{true};
  /// Averaging radii for σ and σ⊥. Part of the parameterisation rather than a
  /// free choice, so overriding them makes the descriptors incomparable with
  /// any solvent ensemble built on the defaults.
  solvent::cosmors::Parameters parameters{};
};

struct ConductorResult {
  solvent::cosmors::Segments segments;
  qm::Wavefunction wavefunction;
  double energy_gas{0.0};
  double energy_conductor{0.0};
  /// Per-element screening energy ½σ_iφ_i on the conductor cavity (Hartree).
  /// Summed this is the variational part of `energy_conductor − energy_gas`;
  /// the remainder is the electronic relaxation cost, which has no per-element
  /// decomposition.
  Vec dielectric_energies;
  double cavity_area{0.0};   ///< Å²
  double cavity_volume{0.0}; ///< Å³
  /// Σ a_i σ_i against −q_solute; the gap is the cavity discretisation error.
  double screening_charge{0.0};
};

/// Build segments from a wavefunction that has already been converged in the
/// ideal-conductor reaction field, with both σ and σ⊥ averaged on `params`.
///
/// Reusing a cached conductor wavefunction here is the whole point of the
/// model: the segment descriptors are solvent independent, so one calculation
/// serves every solvent.
solvent::cosmors::Segments
conductor_segments(const qm::Wavefunction &wavefunction,
                   const solvent::cosmors::Parameters &params = {},
                   double probe_radius_angs = 0.0, int angular_points = 590,
                   bool constrain_charge = true,
                   Vec *dielectric_energies = nullptr,
                   double *cavity_volume_angs3 = nullptr);

/// Converge the SCF in the ideal-conductor reaction field starting from a
/// gas-phase wavefunction, then build the segments.
ConductorResult conductor_profile(const qm::Wavefunction &gas_wavefunction,
                                  const ConductorSettings &settings = {});

/// Settings for a single-molecule openCOSMO-RS solvation free energy.
struct CosmoRSSolvationSettings {
  std::string method{"b3lyp"};
  std::string basis{"def2-svp"};
  bool pure_spherical{true};
  double probe_radius_angs{0.0};
  int angular_points{590};
  /// Constrain the surface charge to -q (Gauss's law for a conductor).
  bool constrain_charge{true};
  double temperature{298.15};
  /// Liquid-phase volume per solute molecule, A^3. Non-positive drops the
  /// reference-state term.
  double liquid_volume{0.0};
  /// Rings in the solute. Negative counts them from the bond graph.
  int num_rings{-1};
};

/// A solvation free energy and the intermediates worth keeping.
struct CosmoRSSolvation {
  solvent::cosmors::SolvationEnergy energy;
  double cavity_area{0.0};   ///< A^2
  double cavity_volume{0.0}; ///< A^3
  int num_rings{0};          ///< as used, whether given or counted
  qm::Wavefunction gas;
  qm::Wavefunction conductor;

  /// The model's solvation free energy, Hartree.
  [[nodiscard]] double total() const { return energy.total(); }
};

/// Solvation free energy of `solute` in a named solvent, end to end.
///
/// Runs the gas-phase SCF, converges it again in the ideal-conductor
/// reaction field, builds the segment descriptors on the model's own
/// averaging radii, loads the solvent's cached segment ensemble and
/// assembles the free energy. This is the whole model in one call; the
/// pieces are available separately for callers that need them.
CosmoRSSolvation
cosmors_solvation_free_energy(const core::Molecule &solute,
                              const std::string &solvent_name,
                              const CosmoRSSolvationSettings &settings = {});

/// The same, computing the solvent's conductor cavity from its geometry
/// instead of loading a cached ensemble. Two SCFs rather than one, but it
/// works for any solvent.
CosmoRSSolvation
cosmors_solvation_free_energy(const core::Molecule &solute,
                              const core::Molecule &solvent,
                              const CosmoRSSolvationSettings &settings = {});

/// Solvent names with a cached segment ensemble, sorted.
std::vector<std::string> available_cosmors_solvents();

} // namespace occ::driver
