#include <cmath>
#include <complex>

#include <occ/core/meshgrid.h>
#include <occ/sht/quadrature.h>
#include <occ/sht/sht.h>
#include <vector>

namespace occ::sht {

int next_power_of_2(int n) {
  int i = 1;
  while (i < n)
    i *= 2;
  return i;
}

int closest_int_with_only_prime_factors_up_to_fmax(int n, int fmax = 7) {
  if (n <= fmax) {
    return n;
  }
  if (fmax < 2) {
    return 0;
  }
  if (fmax == 2) {
    return next_power_of_2(n);
  }

  n -= 2 - (n & 1);
  int f = 2;
  do {
    n += 2;
    f = 2;
    while ((2 * f <= n) && ((n & f) == 0))
      f *= 2; // no divisions for factor 2.
    int k = 3;
    while ((k <= fmax) && (k * f <= n)) {
      while ((k * f <= n) && (n % (k * f) == 0))
        f *= k;
      k += 2;
    }
  } while (f != n);

  int k = next_power_of_2(n); // what is the closest power of 2 ?

  if ((k - n) * 33 < n) {
    return k; // rather choose power of 2 if not too far(3 %)
  }
  return n;
}

SHT::SHT(size_t lm)
    : m_lmax(lm),
      m_nphi(closest_int_with_only_prime_factors_up_to_fmax(2 * lm + 1)),
      m_fft_shape({m_nphi}), m_plm(lm) {
  m_phi = Vec(m_nphi);
  for (size_t i = 0; i < m_nphi; i++) {
    m_phi(i) = (2 * M_PI * i) / m_nphi;
  }

  m_ntheta = m_lmax + 1;
  m_ntheta += (m_ntheta & 1);
  m_ntheta = ((m_ntheta + 7) / 8) * 8;

  std::tie(m_cos_theta, m_weights) = gauss_legendre_quadrature(m_ntheta);
  m_weights.array() /= 2.0;
  m_theta = m_cos_theta.array().acos();
  std::tie(m_theta_grid, m_phi_grid) = occ::core::meshgrid(m_theta, m_phi);
  m_fft_work_array = CVec(m_nphi);
  m_plm_work_array = m_plm.work_array();
}

CVec SHT::analysis_real(const Mat &values) {

  CVec coeffs = CVec::Zero(m_plm_work_array.rows());

  for (int itheta = 0; itheta < m_ntheta; itheta++) {

    double ct = m_cos_theta(itheta);
    double w = m_weights(itheta);

    m_fft_work_array = values.col(itheta);
    pocketfft::r2c(m_fft_shape, m_fft_stride, m_fft_stride, m_fft_axes,
                   pocketfft::FORWARD,
                   reinterpret_cast<const double *>(m_fft_work_array.data()),
                   m_fft_work_array.data(), 4 * M_PI / m_nphi);

    m_plm.evaluate_batch(ct, m_plm_work_array);

    Eigen::Index plm_idx = 0;
    for (int l = 0; l <= m_lmax; l++) {
      double pw = m_plm_work_array(plm_idx) * w;
      coeffs(plm_idx) += m_fft_work_array(0) * pw;
      plm_idx++;
    }

    /*
        because we don't include a phase factor (-1)^m in our
        Associated Legendre Polynomials, we need a factor here.
        which alternates with m and l
    */
    for (int m = 1; m <= m_lmax; m++) {
      int sign = (m & 1) ? -1 : 1;
      for (int l = m; l <= m_lmax; l++) {
        auto pw = sign * m_plm_work_array(plm_idx) * w;
        coeffs(plm_idx) += m_fft_work_array(m) * pw;
        plm_idx++;
      }
    }
  }
  return coeffs;
}

Mat SHT::synthesis_real(const CVec &coeffs) {
  Mat result(m_nphi, m_ntheta);

  for (int itheta = 0; itheta < m_ntheta; itheta++) {
    m_fft_work_array.setZero();
    double ct = m_cos_theta(itheta);

    m_plm.evaluate_batch(ct, m_plm_work_array);

    Eigen::Index plm_idx = 0;
    for (int l = 0; l <= m_lmax; l++) {
      double p = m_plm_work_array(plm_idx);
      m_fft_work_array(0) += coeffs(plm_idx) * p;
      plm_idx++;
    }

    for (int m = 1; m <= m_lmax; m++) {
      int sign = (m & 1) ? -1 : 1;
      for (int l = m; l <= m_lmax; l++) {
        auto p = 2 * sign * m_plm_work_array(plm_idx);
        m_fft_work_array(m) += coeffs(plm_idx) * p;
        plm_idx++;
      }
    }

    // should be able to use a c2r here instead
    pocketfft::c2c(m_fft_shape, m_fft_stride, m_fft_stride, m_fft_axes,
                   pocketfft::BACKWARD, m_fft_work_array.data(),
                   m_fft_work_array.data(), 1.0);

    result.col(itheta) = m_fft_work_array.real();
  }
  return result;
}

CVec SHT::analysis_cplx(const CMat &values) {

  CVec coeffs = CVec::Zero(nlm());

  for (int itheta = 0; itheta < m_ntheta; itheta++) {

    double ct = m_cos_theta(itheta);
    double w = m_weights(itheta);

    m_fft_work_array = values.col(itheta);
    pocketfft::c2c(m_fft_shape, m_fft_stride, m_fft_stride, m_fft_axes,
                   pocketfft::FORWARD, m_fft_work_array.data(),
                   m_fft_work_array.data(), 4 * M_PI / m_nphi);

    m_plm.evaluate_batch(ct, m_plm_work_array);

    Eigen::Index plm_idx = 0;
    for (int l = 0; l <= m_lmax; l++) {
      int l_offset = l * (l + 1);
      double pw = m_plm_work_array(plm_idx) * w;
      coeffs(l_offset) += m_fft_work_array(0) * pw;
      plm_idx++;
    }

    /*
        because we don't include a phase factor (-1)^m in our
        Associated Legendre Polynomials, we need a factor here.
        which alternates with m and l
    */
    for (int m = 1; m <= m_lmax; m++) {
      int sign = (m & 1) ? -1 : 1;
      for (int l = m; l <= m_lmax; l++) {
        int l_offset = l * (l + 1);
        int m_idx_neg = m_nphi - m;
        int m_idx_pos = m;
        auto pw = sign * m_plm_work_array(plm_idx) * w;

        auto rr = m_fft_work_array(m_idx_pos) * pw;
        auto ii = m_fft_work_array(m_idx_neg) * pw;
        if (m & 1)
          ii = -ii;

        coeffs(l_offset - m) += ii;
        coeffs(l_offset + m) += rr;
        plm_idx++;
      }
    }
  }
  return coeffs;
}

CMat SHT::synthesis_cplx(const CVec &coeffs) {
  CMat result(m_nphi, m_ntheta);

  for (int itheta = 0; itheta < m_ntheta; itheta++) {
    m_fft_work_array.setZero();
    double ct = m_cos_theta(itheta);

    m_plm.evaluate_batch(ct, m_plm_work_array);

    Eigen::Index plm_idx = 0;
    for (int l = 0; l <= m_lmax; l++) {
      int l_offset = l * (l + 1);
      double p = m_plm_work_array(plm_idx);
      m_fft_work_array(0) += coeffs(l_offset) * p;
      plm_idx++;
    }

    for (int m = 1; m <= m_lmax; m++) {
      int sign = (m & 1) ? -1 : 1;
      for (int l = m; l <= m_lmax; l++) {
        int l_offset = l * (l + 1);
        int m_idx_neg = m_nphi - m;
        int m_idx_pos = m;
        auto p = sign * m_plm_work_array(plm_idx);
        auto rr = coeffs(l_offset + m) * p;
        auto ii = coeffs(l_offset - m) * p;
        if (m & 1)
          ii = -ii;

        m_fft_work_array(m_idx_neg) += ii;
        m_fft_work_array(m_idx_pos) += rr;
        plm_idx++;
      }
    }

    // should be able to use a c2r here instead
    pocketfft::c2c(m_fft_shape, m_fft_stride, m_fft_stride, m_fft_axes,
                   pocketfft::BACKWARD, m_fft_work_array.data(),
                   m_fft_work_array.data(), 1.0);

    result.col(itheta) = m_fft_work_array.real();
  }
  return result;
}

} // namespace occ::sht
