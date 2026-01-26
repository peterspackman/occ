#pragma once
#include <deque>
#include <occ/core/linear_algebra.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

/// ADIIS (Augmented Direct Inversion in Iterative Subspace)
/// Based on Hu & Yang, J. Chem. Theory Comput. 2010, 6, 2274-2283
/// Uses Augmented Roothaan-Hall energy function for more robust convergence
class ADIIS {
public:
  ADIIS(size_t start = 2, size_t diis_subspace = 8);

  /// Update with new density and Fock matrices, returns extrapolated Fock
  Mat update(SpinorbitalKind kind, const Mat &D, const Mat &F);

  /// Current error estimate
  double error() const { return m_error; }

  /// Reset ADIIS history
  void reset();

private:
  void minimize_coefficients(SpinorbitalKind kind);

  std::deque<Mat> m_density_matrices;
  std::deque<Mat> m_fock_matrices;

  double m_error{1.0};
  size_t m_start{1};
  size_t m_iter{0};
  size_t m_max_subspace_size{8};
  size_t m_nskip{0};
  Vec m_coeffs;
};

} // namespace occ::qm
