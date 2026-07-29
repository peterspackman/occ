#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

/// Split a total electron count into α and β occupations, e.g.
/// `(9, 1) → (5, 4)`. The unpaired count is clamped so β can't go negative.
struct AlphaBetaOccupation {
  double n_alpha{0.0};
  double n_beta{0.0};
};
AlphaBetaOccupation alpha_beta_occupation(double n_elec, double n_unpaired);

struct OrbitalFilling {
  /// Occupation per orbital, each in [0, 1] — one spin channel.
  Vec occupations;
  /// Fermi level (Hartree); zero when no smearing was applied.
  double fermi_level{0.0};
  /// −T·S for this channel (≤ 0): kT · Σ [f ln f + (1−f) ln(1−f)].
  double entropy_energy{0.0};
};

/// Fill `n_electrons` into one spin channel by Fermi-Dirac smearing at
/// electronic temperature `kt` (this is kB·T in Hartree, not T in Kelvin).
///
/// `kt <= 0`, or a full set of orbitals, falls back to aufbau filling — which
/// is also where any gapped system lands, since at the default 300 K
/// kT ≈ 1 mHa leaves occupations integral to machine precision.
OrbitalFilling fermi_filling(double n_electrons, double kt,
                             const Vec &orbital_energies);

} // namespace occ::xtb
