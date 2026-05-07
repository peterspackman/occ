#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/xtb/xtb_result.h>

namespace occ::xtb {

class Gfn2Parameters;

// Backwards-compatible alias for the unified XtbResult.
using SccResult = XtbResult;

struct SccOptions {
  int max_iterations{250};
  double charge_threshold{1e-6};
  double energy_threshold{1e-7};
  double damping_factor{0.4};   // weight on the previous iteration
  double total_charge{0.0};     // net molecular charge (electrons removed)
  int unpaired_electrons{0};    // GFN2 is restricted by default; this is unused for v1
  double electronic_temperature{300.0}; // K, for Fermi smearing
  bool include_dispersion{true};        // add D4 dispersion (EEQ-based for now)
};

// Run a charge-only GFN2 SCC: H0 + isotropic Coulomb, no multipoles, no
// third-order, no dispersion. Useful as a Phase 2 sanity check; full GFN2
// adds those later phases.
SccResult run_charge_only_scc(const std::vector<core::Atom> &atoms,
                              const Gfn2Parameters &params,
                              const SccOptions &opts = {});

// Run a full GFN2 SCC including third-order on-site, anisotropic CAMM
// multipole electrostatics, and on-site polarization. Excludes D4 dispersion
// (Phase 4). All energies are in Hartree.
SccResult run_gfn2_scc(const std::vector<core::Atom> &atoms,
                       const Gfn2Parameters &params,
                       const SccOptions &opts = {});

} // namespace occ::xtb
