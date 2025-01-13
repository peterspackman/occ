#pragma once
#include <deque>
#include <occ/core/linear_algebra.h>
namespace occ::core::diis {

class DIIS {
public:
  DIIS(size_t start = 2, size_t diis_subspace = 20, double damping_factor = 0.0,
       size_t ngroup = 1, size_t ngroup_diis = 1, double mixing_fraction = 0);

  void extrapolate(Mat &x, Mat &error, bool extrapolate_error = false);
  void set_error(double e);
  double error() const;

private:
  double m_error;
  bool m_error_is_set{false};
  size_t m_start{1};
  size_t m_diis_subspace_size{6};
  size_t m_iter{0};
  size_t m_num_group{1};
  size_t m_num_group_diis{1};
  double m_damping_factor{0.0};
  double m_mixing_fraction{0.0};

  Mat m_B; //!< B(i,j) = <ei|ej>

  std::deque<Mat>
      m_x; //!< set of most recent x given as input (i.e. not exrapolated)
  std::deque<Mat> m_errors;       //!< set of most recent errors
  std::deque<Mat> m_extrapolated; //!< set of most recent extrapolated x

  void init();
};

} // namespace occ::core::diis
