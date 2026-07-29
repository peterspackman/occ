#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/xtb/camm.h>

namespace occ::xtb {

/// The quantities the SCC Hamiltonian is built from, bundled so they can be
/// mixed as one vector between iterations.
///
/// Anything left out of the mixer is fed back undamped and free to oscillate,
/// so every input to H belongs here: shell charges always, the shell
/// magnetization when spin-unrestricted, and the CAMM atomic multipoles when
/// the anisotropic terms are on. Unused channels stay empty and cost nothing.
struct SccMixerState {
  Vec shell_charges;
  Vec magnetization;
  CammMoments multipoles;

  /// Allocate the requested channels, zeroed.
  static SccMixerState zero(int n_shells, int n_atoms, bool magnetization,
                            bool multipoles);

  Eigen::Index size() const;
  Vec pack() const;
  void unpack(const Vec &packed);

  /// Move each channel a fraction `1 - factor` of the way toward `fresh`.
  void damp_toward(const SccMixerState &fresh, double factor);

  /// Largest absolute per-element difference across all active channels —
  /// the SCC's charge-convergence metric.
  double max_change(const SccMixerState &other) const;
};

} // namespace occ::xtb
