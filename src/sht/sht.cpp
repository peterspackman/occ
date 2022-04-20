#include <cmath>
#include <complex>
#include <iostream>
#include <occ/core/logger.h>
#include <occ/sht/quadrature.h>
#include <occ/sht/sht.h>
#include <vector>

namespace occ::sht {

SHT::SHT(size_t lm) : l_max(lm), plm(lm), m_nphi(2 * lm + 1), shape({m_nphi}) {
    phi_pts = Vec(m_nphi);
    for (size_t i = 0; i < m_nphi; i++) {
        phi_pts(i) = (2 * M_PI * i) / m_nphi - M_PI;
    }
    m_ntheta = 1;
    while (m_ntheta < m_nphi) {
        m_ntheta *= 2;
    }
    std::tie(cos_theta, weights) = gauss_legendre_quadrature(m_ntheta);
    fft_work_array = CVec(m_nphi);
}

} // namespace occ::sht
