#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

/// Unified GFN2-xTB SCC result. Used by both the molecular and periodic
/// (Γ-only / k-point) drivers. For the periodic case, MO-related fields
/// (`orbital_energies`, `density_matrix`, `overlap_matrix`,
/// `orbital_coefficients`) are reported at the Γ point.
struct XtbResult {
  /// SCC contribution (electronic + isotropic Coulomb + 3rd-order +
  /// AES + on-site polariz, where applicable). Hartree.
  double scc_energy{0.0};
  /// Closed-form repulsion energy (Hartree).
  double repulsion_energy{0.0};
  /// D4 dispersion energy (Hartree). Zero if dispersion is disabled.
  double dispersion_energy{0.0};
  /// scc_energy + repulsion_energy + dispersion_energy (Hartree).
  double total_energy{0.0};

  /// Per-shell partial charges q_shell = ref_occ − Mulliken_pop.
  /// Length = N_shells.
  Vec shell_charges;
  /// Per-atom partial charges q_atom = Σ q_shell over the atom's shells.
  /// Length = N_atoms.
  Vec atomic_charges;

  /// Orbital energies ε at Γ (Hartree). Length = N_basis.
  Vec orbital_energies;
  /// Orbital occupations n_i (0..2 for closed shell). Length = N_basis.
  Vec orbital_occupations;
  /// Density matrix P at Γ (closed-shell, summed over both spins).
  Mat density_matrix;
  /// Overlap matrix S at Γ (cached for downstream property analysis).
  Mat overlap_matrix;
  /// Orbital coefficients C at Γ (closed-shell).
  Mat orbital_coefficients;

  /// Number of SCC iterations actually run.
  int n_iterations{0};
  /// True if the SCC converged within the iteration / threshold limits.
  bool converged{false};
};

} // namespace occ::xtb
