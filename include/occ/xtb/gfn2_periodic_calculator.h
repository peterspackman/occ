#pragma once
#include <occ/core/atom.h>
#include <occ/gto/shell.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/periodic.h>
#include <occ/xtb/periodic_gamma.h>
#include <occ/xtb/scc.h>

namespace occ::xtb {

struct PeriodicSccOptions {
  int max_iterations{250};
  double charge_threshold{1e-6};
  double energy_threshold{1e-7};
  double damping_factor{0.4};
  double total_charge{0.0};       // net charge per primitive cell
  // Real-space cutoff (Bohr) for periodic CN, repulsion, and per-T AO matrix
  // blocks. The Ewald γ uses its own (separate) cutoffs.
  double real_cutoff{20.0};
  // Override Ewald α (1/Bohr); 0 → auto-pick.
  double ewald_alpha{0.0};
  // Override residual γ-1/R cutoff (Bohr); 0 → 60 Bohr default.
  double ewald_residual_cutoff{0.0};
};

struct PeriodicSccResult {
  double scc_energy{0.0};      // electronic + isotropic Coulomb (Hartree)
  double repulsion_energy{0.0};
  double total_energy{0.0};    // scc + repulsion (no dispersion in v1)
  Vec shell_charges;           // per-shell, central cell
  Vec atomic_charges;          // per-atom, central cell
  Vec orbital_energies;        // ε at Γ (Hartree)
  Vec orbital_occupations;     // n_i (0..2)
  Mat density_matrix;          // P at Γ (real, closed-shell)
  Mat overlap_matrix;          // S at Γ
  Mat orbital_coefficients;    // C at Γ
  int n_iterations{0};
  bool converged{false};
};

// Γ-point GFN2 SCC for a 3D periodic system. Charge-only (no multipole CAMM
// or anisotropic ES) and no dispersion in this v1; multipoles + periodic D4
// come in subsequent phases.
//
// Periodic ingredients used:
//   - real-space CN, repulsion (sums over translations within real_cutoff)
//   - per-T overlap and H0 blocks → Bloch-summed at Γ
//   - shell-resolved γ via Ewald (`periodic_klopman_ohno_gamma`)
//
// For a sufficiently large unit cell, this reduces to the molecular charge-
// only GFN2 SCC.
PeriodicSccResult
run_charge_only_periodic_scc(const PeriodicSystem &sys,
                              const Gfn2Parameters &params,
                              const PeriodicSccOptions &opts = {});

} // namespace occ::xtb
