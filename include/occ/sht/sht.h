#pragma once
#include <occ/3rdparty/pocketfft.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/sht/legendre.h>

namespace occ::sht {

class SHT {
  public:
    using Complex = std::complex<double>;

    SHT(size_t lm);
    inline auto ntheta() const { return m_ntheta; }
    inline auto nphi() const { return m_nphi; }

    inline auto nlm() const { return (m_lmax + 1) * (m_lmax + 1); }

    inline auto nplm() const { return (m_lmax + 1) * (m_lmax + 2) / 2; }

    const auto &grid_phi() const { return m_phi_grid; }
    const auto &grid_theta() const { return m_theta_grid; }

    CVec analysis_real(const Mat &);
    Mat synthesis_real(const CVec &);

    CVec analysis_cplx(const CMat &);
    CMat synthesis_cplx(const CVec &);

    template <typename F> CMat values_on_grid_complex(F &f) {
        CMat values(m_nphi, m_ntheta);
        for (size_t i = 0; i < m_nphi; i++) {
            for (size_t j = 0; j < m_ntheta; j++) {
                values(i, j) = f(m_theta(j), m_phi(i));
            }
        }
        return values;
    }

    template <typename F> Mat values_on_grid_real(F &f) {
        Mat values(m_nphi, m_ntheta);
        for (size_t i = 0; i < m_nphi; i++) {
            for (size_t j = 0; j < m_ntheta; j++) {
                values(i, j) = f(m_theta(j), m_phi(i));
            }
        }
        return values;
    }

  private:
    size_t m_lmax{0};
    size_t m_nphi{0}, m_ntheta{0};
    Mat m_phi_grid;
    Mat m_theta_grid;
    Vec m_weights;
    Vec m_theta;
    Vec m_phi;
    Vec m_cos_theta;
    CVec m_fft_work_array;
    Vec m_plm_work_array;

    pocketfft::shape_t m_fft_shape;
    pocketfft::shape_t m_fft_axes{0};
    pocketfft::stride_t m_fft_stride{sizeof(Complex)};
    AssocLegendreP m_plm;
};

} // namespace occ::sht
