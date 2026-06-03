#pragma once
#include <occ/qm/cc/integrals.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

struct CCSDResult {
  double e_corr{0.0};
  Eigen::Tensor<double, 2> t1; ///< (nocc x nvir)
  Eigen::Tensor<double, 4> t2; ///< (nocc x nocc x nvir x nvir)
  int iterations{0};
  bool converged{false};
};

struct CCSDOptions {
  int max_cycle{100};
  double tol{1e-9}; ///< convergence on the correlation energy
  bool diis{true};
};

/// Restricted (closed-shell, spin-adapted) CCSD against a canonical reference.
/// Backend-agnostic: `eris` may be exact / DF / THC. Returns the correlation
/// energy and converged t1/t2 amplitudes.
CCSDResult ccsd(const CCIntegrals &eris, const CCSDOptions &opts = {});

/// CCSD correlation energy for given amplitudes (exposed for testing).
double ccsd_energy(const Eigen::Tensor<double, 2> &t1,
                   const Eigen::Tensor<double, 4> &t2, const CCIntegrals &eris);

} // namespace occ::qm::cc
