#pragma once
#include <occ/qm/cc/integrals.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm::cc {

/// Restricted closed-shell perturbative triples (T) correction.
///
/// Port of the thc_cct reference rtriples.py (PySCF ccsd_t_slow, JCP 94, 442
/// (1991)). Memory-frugal: it loops over virtual triples (a>=b>=c) and forms
/// only O(nocc^3) intermediates per triple -- the full O(o^3 v^3) triples array
/// is never materialised. The only 3-virtual integral needed is `ovvv`, which
/// is supplied by the (exact / DF / THC) backend, so (T) works for all three.
///
/// Returns the (T) energy correction for the given converged CCSD amplitudes.
double ccsd_t(const Eigen::Tensor<double, 2> &t1,
              const Eigen::Tensor<double, 4> &t2, const CCIntegrals &eris);

} // namespace occ::qm::cc
