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
  // Include the CAMM anisotropic electrostatics + on-site polarization terms
  // (xtb's "AES"). True for full GFN2; false for the charge-only fast path.
  bool include_multipoles{true};
  // Include native DFT-D4 dispersion (lattice-summed BJ damping over the
  // translations within `disp_cutoff`).
  bool include_dispersion{true};
  // Real-space cutoff (Bohr) for the lattice-summed D4 BJ pair sum. 60 Bohr
  // gives ~ µHa precision for typical molecular crystals; tighter cutoffs are
  // fine for production speed.
  double disp_cutoff{60.0};
  // Per-quantity real-space cutoffs (Bohr). CN's exponential count decays
  // fast — 25 matches tblite's `default_cutoff` in ncoord/gfn.f90.
  // Repulsion uses an exp(-α·r) form, fully converged by 30 Bohr. AO
  // multipole / H0 / S blocks decay as exp(-α·r²) but build_h0's per-T
  // sum has long-range tails on dense π-stacked crystals (anthracene,
  // triazine) — 20 Bohr is the empirical convergence threshold; tighter
  // values risk ~tens of mHa errors. The Ewald γ uses its own cutoffs.
  double cn_cutoff{25.0};
  double rep_cutoff{30.0};
  double ao_cutoff{20.0};
  // Override Ewald α (1/Bohr); 0 → auto-pick.
  double ewald_alpha{0.0};
  // Override residual γ-1/R cutoff (Bohr); 0 → 60 Bohr default.
  double ewald_residual_cutoff{0.0};
};

struct PeriodicSccResult {
  double scc_energy{0.0};      // electronic + isotropic Coulomb (Hartree)
  double repulsion_energy{0.0};
  double dispersion_energy{0.0};
  double total_energy{0.0};    // scc + repulsion + dispersion
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

// Γ-point GFN2 SCC for a 3D periodic system. Charge-only or full GFN2
// (including CAMM multipoles, anisotropic ES, on-site polarization) selected
// by `opts.include_multipoles`. No dispersion in this v1.
//
// Periodic ingredients used:
//   - real-space CN, repulsion (sums over translations within real_cutoff)
//   - per-T overlap and H0 blocks → Bloch-summed at Γ
//   - shell-resolved γ via Ewald (`periodic_klopman_ohno_gamma`)
//   - multipole pair sum: real-space lattice sum (gab3/gab5 inline, no Ewald)
//
// For a sufficiently large unit cell, this reduces to the molecular GFN2 SCC.
PeriodicSccResult
run_charge_only_periodic_scc(const PeriodicSystem &sys,
                              const Gfn2Parameters &params,
                              const PeriodicSccOptions &opts = {});

// k-point sampled GFN2 SCC. For each iteration the SCC builds H(k) and S(k)
// at every k via Bloch sum of the per-T blocks, solves the complex Hermitian
// generalized eigenproblem, and accumulates the density and Mulliken charges
// across the k-grid with the given weights.
//
// Charge-only (no multipoles in this v1 — multipoles need a different
// density-partition strategy under k-sampling that we'll wire later).
//
// `kpoints` is a Monkhorst-Pack-style mesh (use `monkhorst_pack_grid` to
// build). For a single Γ-point ({n1, n2, n3} = {1, 1, 1}), this matches the
// real-arithmetic Γ-only path of `run_charge_only_periodic_scc` to ~1e-9.
PeriodicSccResult
run_periodic_scc_kpoints(const PeriodicSystem &sys,
                          const Gfn2Parameters &params,
                          const std::vector<struct KPoint> &kpoints,
                          const PeriodicSccOptions &opts = {});

} // namespace occ::xtb
