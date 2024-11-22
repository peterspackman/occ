#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/sht/spherical_harmonics.h>

namespace occ::descriptors {

class Steinhardt {
public:
  Steinhardt(size_t lmax);

  Vec compute_q(Eigen::Ref<const Mat3N> positions);

  Vec compute_w(Eigen::Ref<const Mat3N> positions);

  CVec compute_qlm(Eigen::Ref<const Mat3N> positions);

  void precompute_wigner3j_coefficients();

  inline size_t size() const { return m_lmax + 1; }
  inline size_t nlm() const { return m_harmonics.nlm(); }

  Vec compute_averaged_q(Eigen::Ref<const Mat3N> positions,
                         double radius = 6.0);
  Vec compute_averaged_w(Eigen::Ref<const Mat3N> positions,
                         double radius = 6.0);

private:
  struct Wigner3jCache {
    int l;
    int m1, m2, m3;
    double coeff;
  };

  size_t m_lmax;
  sht::SphericalHarmonics m_harmonics;
  std::vector<Wigner3jCache> m_wigner_coefficients;

  Vec m_q;
  Vec m_w;
  CVec m_qlm, m_ylm;
};

} // namespace occ::descriptors
