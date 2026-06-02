#pragma once
#include <occ/qm/cc/uintegrals.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

/// Perturbative (T) correction for spin-adapted unrestricted CCSD, evaluated
/// natively from the spin-blocked UCCIntegrals (no spin-orbital integral build).
/// Port of PySCF uccsd_t_slow.py (4 spin cases aaa/bbb/baa/bba); a canonical UHF
/// reference is assumed so the f_vo disconnected terms drop.
double uccsd_t(const UCCIntegrals &e, const Eigen::Tensor<double, 2> &t1a,
               const Eigen::Tensor<double, 2> &t1b,
               const Eigen::Tensor<double, 4> &t2aa,
               const Eigen::Tensor<double, 4> &t2ab,
               const Eigen::Tensor<double, 4> &t2bb);

} // namespace occ::qm::cc
