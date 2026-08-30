#pragma once
#include <occ/cg/solvation_data.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/core/molecule.h>
#include <occ/driver/sigma_driver.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/sigma_solvation.h>
#include <string>
#include <vector>

namespace occ::driver {

struct SigmaSolvationSettings {
  std::string method{"b3lyp"};
  std::string basis{"def2-tzvp"};
  bool pure_spherical{true};
  int angular_points{590};
  double temperature{298.15};
  solvent::sigma::Model model{solvent::sigma::Model::CosmoSac2010};
};

/// Per-monomer solvation surfaces from a σ-potential model, plus the
/// per-contact descriptors that come with them.
///
/// `surfaces` carries the residual solvation energy per surface element in
/// the `occ::cg` shape, so `SolventSurfacePartitioner` consumes it unchanged.
/// `reorganisation` and `hbond_area` are per-element and partition the same
/// way, giving each neighbour contact a descriptor alongside its energy.
struct SigmaSolvationResult {
  std::vector<cg::SolvationData> surfaces;
  std::vector<qm::Wavefunction> wavefunctions; ///< ideal-conductor
  std::vector<Vec> reorganisation;             ///< Hartree, per element
  std::vector<Vec> hbond_area;                 ///< Å², per element
};

/// Build solvation surfaces for each molecule in `solvent`.
///
/// The conductor wavefunction is solvent independent, so it is computed once
/// per molecule and cached as `<basename>_<i>_conductor.owf.json`. Changing
/// solvent then costs one apparent-surface-charge solve and a contraction,
/// not another SCF.
///
/// The σ-potential supplies the *residual* term only; no absolute solvation
/// free energy is claimed, and the CDS branch is left empty.
SigmaSolvationResult
sigma_solvation(const std::string &basename,
                const std::vector<core::Molecule> &molecules,
                const std::vector<qm::Wavefunction> &gas_wavefunctions,
                const SolventSpec &solvent,
                const SigmaSolvationSettings &settings = {});

/// The same, using the openCOSMO-RS kernel and its cached `.rsseg` solvent
/// ensembles.
///
/// Three channels are produced, all additive over surface elements:
/// `dielectric` (gas to ideal conductor), `residual` (conductor to solvent),
/// and `cavity` (the per-atom `τ_α A_α` term). The remaining terms of the
/// openCOSMO-RS solvation free energy — ring, reference state and the
/// constant η — are per-molecule and have no surface-element home; they are
/// the same for a bulk and a surface molecule, so they cancel in the
/// attachment-energy difference cg forms and are deliberately omitted here.
/// The absolute solvation free energy is what `occ sigma` reports, not this.
CGSolvationResult
opencosmors_solvation(const std::string &basename,
                      const std::vector<core::Molecule> &molecules,
                      const std::vector<qm::Wavefunction> &gas_wavefunctions,
                      const SolventSpec &solvent,
                      const SigmaSolvationSettings &settings = {});

} // namespace occ::driver
