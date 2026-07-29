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
  // Nα − Nβ. Non-zero switches the SCC to the spin-unrestricted path
  // (separate α/β densities). Must have the same parity as the electron count.
  int unpaired_electrons{0};
  // Scale on the on-site spin-polarization constants W. 1 (default) is the
  // spin-polarized GFN2 treatment; 0 reproduces the common-Fock open-shell
  // result of plain `xtb --uhf` (α and β share one Hamiltonian and differ
  // only in their occupations).
  double spin_polarization{1.0};
  // Force the unrestricted path even for a closed-shell electron count.
  // Without a symmetry-broken guess this converges to the restricted answer;
  // it exists so the two code paths can be compared directly.
  bool force_unrestricted{false};
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
