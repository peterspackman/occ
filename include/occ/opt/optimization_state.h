#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::opt {

/// Thresholds on per-atom quantities: the norm of each atom's gradient and
/// of its displacement in the proposed step. Defaults are ORCA's NormalOpt
/// values.
struct ConvergenceCriteria {
  double gradient_max = 3e-4; // Hartree/Bohr
  double gradient_rms = 1e-4; // Hartree/Bohr
  double step_max = 4e-3;     // Bohr
  double step_rms = 2e-3;     // Bohr

  // Energy change between steps (Hartree), used only on request
  double energy_change = 5e-6;
  bool use_energy_criterion = false;

  int max_iterations = 100;
};

/// Checks and logs the criteria for the gradient (3 x N, Hartree/Bohr), the
/// per-atom displacements of the proposed step (Bohr) and the energy change
/// since the previous point (ignored when check_energy is false). Returns
/// true when every active criterion is met.
bool check_convergence(const Mat3N &gradient, const Vec &displacements,
                       double energy_change, bool check_energy,
                       const ConvergenceCriteria &criteria, int step);

} // namespace occ::opt
