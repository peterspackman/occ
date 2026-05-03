#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

class Gfn2Parameters;

struct SccResult {
  double scc_energy{0.0};      // electronic + isotropic Coulomb (Hartree)
  double repulsion_energy{0.0};
  double dispersion_energy{0.0};
  double total_energy{0.0};    // scc + repulsion + dispersion
  Vec shell_charges;           // q_shell = ref_occ - mulliken_pop (length = N_shells)
  Vec atomic_charges;          // q_atom = sum over shells (length = N_atoms)
  Vec orbital_energies;        // ε in Hartree
  Vec orbital_occupations;     // n_i (0..2)
  Mat density_matrix;          // P (closed-shell, ∑ over both spins)
  Mat overlap_matrix;          // S (cached for downstream multipole work)
  Mat orbital_coefficients;    // C (closed-shell)
  int n_iterations{0};
  bool converged{false};
};

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
