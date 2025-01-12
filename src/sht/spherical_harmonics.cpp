#include <occ/sht/spherical_harmonics.h>

namespace occ::sht {

SphericalHarmonics::SphericalHarmonics(size_t lm, bool phase)
    : m_phase{phase}, m_lmax{lm}, m_plm_evaluator(lm),
      m_plm(m_plm_evaluator.work_array()) {}

CVec SphericalHarmonics::evaluate(double theta, double phi) {

  CVec result(nlm());
  evaluate(theta, phi, result);
  return result;
}

void SphericalHarmonics::evaluate(double theta, double phi,
                                  Eigen::Ref<CVec> result) {

  using cplx = std::complex<double>;
  double ct = std::cos(theta);
  m_plm_evaluator.evaluate_batch(ct, m_plm);

  Eigen::Index plm_idx = 0;
  for (int l = 0; l <= m_lmax; l++) {
    int l_offset = l * (l + 1);
    result(l_offset) = m_plm(plm_idx);
    plm_idx++;
  }

  const cplx c = std::exp(cplx(0, phi));
  cplx cm = c;
  for (int m = 1; m <= m_lmax; m++) {
    int sign = (m_phase & (m & 1)) ? -1 : 1;
    for (int l = m; l <= m_lmax; l++) {
      int l_offset = l * (l + 1);
      cplx rr = cm;
      cplx ii = std::conj(rr);
      rr = sign * m_plm(plm_idx) * rr;
      ii = sign * m_plm(plm_idx) * ii;
      if (m & 1)
        ii = -ii;
      result(l_offset - m) = ii;
      result(l_offset + m) = rr;
      plm_idx++;
    }
    cm *= c;
  }
}

CVec SphericalHarmonics::evaluate(Eigen::Ref<const Vec3> pos) {
  CVec result(nlm());
  evaluate(pos, result);
  return result;
}

void SphericalHarmonics::evaluate(Eigen::Ref<const Vec3> pos,
                                  Eigen::Ref<CVec> result) {
  constexpr double epsilon = 1e-12;
  using cplx = std::complex<double>;
  double ct = pos.z();
  m_plm_evaluator.evaluate_batch(ct, m_plm);

  const double st =
      (std::abs(1.0 - ct) > epsilon) ? std::sqrt(1.0 - ct * ct) : 0.0;

  Eigen::Index plm_idx = 0;
  for (int l = 0; l <= m_lmax; l++) {
    int l_offset = l * (l + 1);
    result(l_offset) = m_plm(plm_idx);
    plm_idx++;
  }

  const cplx c((st > epsilon) ? (pos.x() / st) : 0.0,
               (st > 1e-12) ? (pos.y() / st) : 0.0);
  cplx cm = c;
  for (int m = 1; m <= m_lmax; m++) {
    int sign = (m_phase & (m & 1)) ? -1 : 1;
    for (int l = m; l <= m_lmax; l++) {
      int l_offset = l * (l + 1);
      cplx rr = cm;
      cplx ii = std::conj(rr);
      rr = sign * m_plm(plm_idx) * rr;
      ii = sign * m_plm(plm_idx) * ii;
      if (m & 1)
        ii = -ii;
      result(l_offset - m) = ii;
      result(l_offset + m) = rr;
      plm_idx++;
    }
    cm *= c;
  }
}

} // namespace occ::sht
