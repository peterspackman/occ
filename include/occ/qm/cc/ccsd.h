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
  /// if > 0, converge instead on max |R|, R = D (t_new - t_old) the amplitude
  /// residual (ORCA's MDCI STol)
  double residual_tol{0.0};
  bool diis{true};
};

/// Restricted (closed-shell, spin-adapted) CCSD against a canonical reference.
/// Backend-agnostic: `eris` may be exact / DF / THC. Returns the correlation
/// energy and converged t1/t2 amplitudes.
CCSDResult ccsd(const CCIntegrals &eris, const CCSDOptions &opts = {});

/// CCSD correlation energy for given amplitudes (exposed for testing).
double ccsd_energy(const Eigen::Tensor<double, 2> &t1,
                   const Eigen::Tensor<double, 4> &t2, const CCIntegrals &eris);

/// Moller-Plesset correlation energies through fourth order (canonical RHF).
struct MPnEnergies {
  double e2{0.0}, e3{0.0};
  double e4_singles{0.0}, e4_doubles{0.0}, e4_triples{0.0}, e4_quadruples{0.0};
  double mp3() const { return e2 + e3; }
  double mp4_sdq() const {
    return mp3() + e4_singles + e4_doubles + e4_quadruples;
  }
  double mp4() const { return mp4_sdq() + e4_triples; }
};

/// MP2, MP3 and MP4(SDQ / SDTQ) from the CCSD residual: with t1 = 0 the doubles
/// residual is g + L(t) + Q(t), so the first-order amplitudes t give
/// E3 = <t~|L>, E4(D) = <L~|L/D>, E4(S) = 2 |S|^2/D, E4(Q) = <t~|Q>, and E4(T)
/// is ccsd_t on t (with triples). Costs two CCSD residuals plus (T).
MPnEnergies mpn_energies(const CCIntegrals &eris, bool triples = true);

} // namespace occ::qm::cc
