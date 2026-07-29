#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/xtb/solvation_interface.h>
#include <optional>

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

  /// Orbital energies ε at Γ (Hartree), α channel when unrestricted.
  Vec orbital_energies;
  /// Orbital occupations: 0..2 restricted, 0..1 for the α channel otherwise.
  Vec orbital_occupations;
  /// Density matrix P at Γ, always summed over both spins — so Mulliken,
  /// CAMM and bond-order consumers work unchanged either way.
  Mat density_matrix;
  /// Overlap matrix S at Γ (cached for downstream property analysis).
  Mat overlap_matrix;
  /// Orbital coefficients C at Γ, α channel when unrestricted.
  Mat orbital_coefficients;

  // --- Open shell: empty / zero unless the SCC ran spin-unrestricted -------

  bool unrestricted{false};
  /// Nα − Nβ (multiplicity − 1).
  int num_unpaired_electrons{0};
  /// On-site spin-polarization energy ½ mᵀWm (≤ 0), included in `scc_energy`.
  double spin_energy{0.0};
  /// Fermi-smearing free-energy term −T·S (≤ 0), included in `scc_energy`.
  /// Vanishes for any system with a gap ≫ kT.
  double electronic_entropy_energy{0.0};
  /// Per-shell magnetization, Mulliken pop(α) − pop(β).
  Vec shell_magnetization;
  /// Per-atom spin populations; sums to Nα − Nβ.
  Vec atomic_magnetization;
  /// Spin-resolved densities; `density_matrix` is their sum.
  Mat density_matrix_alpha;
  Mat density_matrix_beta;
  Mat orbital_coefficients_beta;
  Vec orbital_energies_beta;
  Vec orbital_occupations_beta;

  /// Number of SCC iterations actually run.
  int n_iterations{0};
  /// True if the SCC converged within the iteration / threshold limits.
  bool converged{false};

  /// Per-element solvation surfaces (Phase 7D). Populated only when an
  /// `XtbSolvationModel` is attached that exposes surface data — CPCM-X
  /// fills `coulomb`; SMD fills both `coulomb` and `cds`; NullSolvationModel
  /// leaves this empty.
  std::optional<SolvationSurfaces> solvation_surfaces;
};

} // namespace occ::xtb
