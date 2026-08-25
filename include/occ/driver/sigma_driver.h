#pragma once
#include <occ/qm/wavefunction.h>
#include <occ/solvent/sigma_kernel.h>
#include <occ/solvent/sigma_profile.h>
#include <string>

namespace occ::driver {

/// Settings for the ideal-conductor COSMO calculation a σ-profile is built
/// from. The averaging convention comes from `model`, since it is part of the
/// parameterisation rather than a free choice.
struct SigmaProfileSettings {
  std::string method{"b3lyp"};
  std::string basis{"def2-tzvp"};
  bool pure_spherical{true};
  double probe_radius_angs{0.0};
  int angular_points{590};
  solvent::sigma::Model model{solvent::sigma::Model::CosmoSac2010};
};

struct ConductorResult {
  solvent::sigma::Segments segments;
  qm::Wavefunction wavefunction;
  double energy_gas{0.0};
  double energy_conductor{0.0};
  double cavity_area{0.0};   ///< Å²
  double cavity_volume{0.0}; ///< Å³
  /// Σ a_i σ_i against −q_solute; the gap is the cavity discretisation error.
  double screening_charge{0.0};
};

/// Build averaged, H-bond-classified segments from a wavefunction that has
/// already been converged in the ideal-conductor reaction field.
///
/// Reusing a cached conductor wavefunction here is the whole point of the
/// model: the σ-profile is solvent independent, so one calculation serves
/// every solvent.
solvent::sigma::Segments
conductor_segments(const qm::Wavefunction &wavefunction,
                   const solvent::sigma::Parameters &params,
                   double probe_radius_angs = 0.0,
                   int angular_points = 590);

/// Converge the SCF in the ideal-conductor reaction field starting from a
/// gas-phase wavefunction, then build the segments.
ConductorResult conductor_profile(const qm::Wavefunction &gas_wavefunction,
                                  const SigmaProfileSettings &settings = {});

} // namespace occ::driver
