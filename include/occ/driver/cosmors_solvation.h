#pragma once
#include <occ/cg/solvation_data.h>
#include <occ/core/molecule.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/driver/cosmors_driver.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::driver {

struct CosmoRSSettings {
  std::string method{"b3lyp"};
  std::string basis{"def2-svp"};
  bool pure_spherical{true};
  int angular_points{590};
  /// Solvent probe radius used to build the cavity, Angstrom. Zero is the
  /// COSMO convention: the cavity is the scaled van der Waals surface.
  double probe_radius_angs{0.0};
  double temperature{298.15};
  /// Condensed-phase volume per solute molecule, A^3, for the reference-state
  /// term. Zero leaves that term out.
  double volume_per_molecule{0.0};
};

/// Build openCOSMO-RS solvation surfaces for each molecule in `solvent`.
///
/// The conductor wavefunction is solvent independent, so it is computed once
/// per molecule and cached as `<basename>_<i>_conductor.owf.json`. Changing
/// solvent then costs one apparent-surface-charge solve and a contraction,
/// not another SCF.
///
/// Three of the model's terms are additive over surface elements and become
/// cg channels: `dielectric` (gas to ideal conductor), `residual` (conductor
/// to solvent), and `cavity` (the per-atom `τ_α A_α` term). The rest —
/// combinatorial, ring, reference state and the constant η — are per-molecule
/// with no surface-element home, so they carry no channel, but they are still
/// added into `total_solvation_energy` so the reported figure is the model's
/// whole solvation free energy. They are identical for a bulk and a surface
/// molecule, so they cancel in the attachment-energy difference cg forms.
CGSolvationResult cosmors_solvation(
    const std::string &basename, const std::vector<core::Molecule> &molecules,
    const std::vector<qm::Wavefunction> &gas_wavefunctions,
    const SolventSpec &solvent, const CosmoRSSettings &settings = {});

} // namespace occ::driver
