#pragma once
#include <occ/core/diis.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

class EDIIS {
public:
  EDIIS(size_t start = 2, size_t diis_subspace = 8);

  Mat update(SpinorbitalKind kind, const Mat &D, const Mat &F, double e);
  double error() const { return m_error; }
  void reset();

private:
  void minimize_coefficients(SpinorbitalKind kind);
  std::deque<Mat> m_density_matrices;
  std::deque<Mat> m_fock_matrices;
  std::deque<double> m_energies;

  double m_error{1.0};
  size_t m_start{1};
  size_t m_iter{0};
  size_t m_max_subspace_size{8};
  size_t m_nskip{0};
  Vec m_coeffs;
};

} // namespace occ::qm
