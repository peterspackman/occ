#pragma once
#include <occ/core/diis.h>

namespace occ::qm {

class EDIIS {
public:
  EDIIS(size_t start = 2, size_t diis_subspace = 20);

  Mat update(const Mat &D, const Mat &F, double e);
  double error() const { return m_error; }

private:
  void minimize_coefficients();
  std::deque<Mat> m_density_matrices;
  std::deque<Mat> m_fock_matrices;
  std::deque<double> m_energies;

  double m_error{1.0};
  size_t m_start{1};
  size_t m_iter{0};
  size_t m_diis_subspace_size{10};
  size_t m_nskip{0};
  Vec m_previous_coeffs;
};

} // namespace occ::qm
