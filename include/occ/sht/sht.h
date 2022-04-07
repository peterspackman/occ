#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/3rdparty/pocketfft.h>

namespace occ::sht {

std::pair<Vec, Vec> gauss_legendre_quadrature(int N);
void gauss_legendre_quadrature(Vec &roots, Vec &weights, int N);

class AssocLegendreP {
public:
    AssocLegendreP(size_t lm); 
    double operator()(size_t l, size_t m, double x) const; 

    static double amm(size_t m);
    static double alm(size_t l, size_t m); 
    static double blm(size_t l, size_t m);

private:

    size_t l_max{0};
    Mat m_a;
    Mat m_b;
};


class SHT {
public:
    using Complex = std::complex<double>;

    SHT(size_t lm);
    inline auto ntheta() const { return m_ntheta; }
    inline auto nphi() const { return m_nphi; }
    static inline size_t idx_c(int l, int m) { return l * (l + 1) + m; }
    static inline size_t idx_r(int l, int m, int lmax) {
	// should be an explicit representation of this that's convenient but a loop will do
	size_t result = 0;
	bool mfound = false;
	for(size_t mm = 0; mm <= std::abs(m); mm++) {
	    mfound = (mm == m);
	    for(size_t ll = mm; ll <= lmax; ll++) {
		if(mfound && ll == l) break;
		result++;
	    }
	}
	return result;
    }

    template<typename F>
    CVec analysis_real(F& f) {
	CVec coeffs = CVec::Zero((l_max + 1) * (l_max + 2) / 2);
	for(size_t itheta = 0; itheta < weights.size(); itheta++) {
	    const double theta = std::acos(cos_theta(itheta));
	    const double ct = cos_theta[itheta];
	    double w = weights[itheta] / 2.0;
	    for(size_t j = 0; j < m_nphi; j++) {
		fft_work_array(j) = f(theta, phi_pts(j));
	    }

	    pocketfft::r2c(shape, stride, stride, axes, pocketfft::FORWARD,
			   reinterpret_cast<const double*>(fft_work_array.data()), fft_work_array.data(), 4 * M_PI / m_nphi);

	    size_t coeff_idx = 0;
	    for(int m = 0; m <= static_cast<int>(l_max); m++) {
		// m == 0 case
		for(int l = m; l <= static_cast<int>(l_max); l++) {
		    double p = plm(l, m, ct);

		    // this conjugate is to match output from shtns, not sure where it comes from?
		    if(m & 1) {
			coeffs(coeff_idx) += fft_work_array(m) * p * w;
		    }
		    else {
			coeffs(coeff_idx) += std::conj(fft_work_array(m)) * p * w;
		    }
		    coeff_idx++;
		}
	    }
	}
	return coeffs;
    }


    template<typename F>
    CVec analysis_complex(F& f) {
	CVec coeffs = CVec::Zero((l_max + 1) * (l_max + 1));
	for(size_t itheta = 0; itheta < weights.size(); itheta++) {
	    const double theta = std::acos(cos_theta(itheta));
	    const double ct = cos_theta[itheta];
	    double w = weights[itheta] / 2.0;
	    for(size_t j = 0; j < m_nphi; j++) {
		fft_work_array(j) = f(theta, phi_pts(j));
	    }

	    pocketfft::c2c(shape, stride, stride, axes, pocketfft::FORWARD,
			   fft_work_array.data(), fft_work_array.data(), 4 * M_PI / m_nphi);

	    for(int l = 0; l <= static_cast<int>(l_max); l++) {
		// m == 0 case
		const int l_offset = l * (l + 1);
		double pl0 = plm(l, 0, cos_theta[itheta]);
		coeffs[l_offset] += fft_work_array[0] * pl0 * w;
		for(int m = 1; m <= l; m++) {
		    // do both m+ and m- at the same time (same Plm)
		    size_t m_idx_neg = m_nphi - m;
		    size_t m_idx_pos = m;
		    double p = plm(l, m, ct);

		    if(m & 1) {
			coeffs(l_offset - m) += fft_work_array(m_idx_neg) * p * w;
			coeffs(l_offset + m) += fft_work_array(m_idx_pos) * p * w;
		    }
		    else {
			coeffs(l_offset - m) += std::conj(fft_work_array(m_idx_neg)) * p * w;
			coeffs(l_offset + m) += std::conj(fft_work_array(m_idx_pos)) * p * w;
		    }
		}
	    }
	}
	return coeffs;
    }

    CMat synthesis_complex(const CVec &coeffs) {
	CMat values(m_ntheta, m_nphi);
	for(size_t itheta = 0; itheta < weights.size(); itheta++) {
	    double theta = std::acos(cos_theta[itheta]);
	    double w = weights[itheta];
	    fft_work_array.setZero();
	    for(int l = 0; l <= static_cast<int>(l_max); l++) {
		// m == 0 case
		const int l_offset = l * (l + 1);
		double pl0 = plm(l, 0, cos_theta[itheta]);
		fft_work_array(0) += coeffs[l_offset] * pl0;
		for(int m = 1; m <= l; m++) {
		    // do both m+ and m- at the same time (same Plm)
		    int m_idx_neg = m_nphi - m;
		    int m_idx_pos = m;
		    double p = plm(l, m, cos_theta[itheta]);

		    if(m & 1) {
			fft_work_array(m_idx_neg) += coeffs(l_offset - m) * p;
			fft_work_array(m_idx_pos) += coeffs(l_offset + m) * p;
		    }
		    else {
			fft_work_array(m_idx_neg) += std::conj(coeffs(l_offset - m)) * p;
			fft_work_array(m_idx_pos) += std::conj(coeffs(l_offset + m)) * p;
		    }
		}
	    }
	    pocketfft::c2c(shape, stride, stride, axes, pocketfft::BACKWARD,
			   fft_work_array.data(), fft_work_array.data(), 1.0);
	    values.row(itheta) = fft_work_array;
	}
	return values;
    }

    Mat synthesis_real(const CVec &coeffs) {
	Mat values(m_ntheta, m_nphi);
	for(size_t itheta = 0; itheta < weights.size(); itheta++) {
	    double theta = std::acos(cos_theta[itheta]);
	    double w = weights[itheta];
	    fft_work_array.setZero();
	    size_t coeff_idx = 0;
	    for(int m = 0; m <= static_cast<int>(l_max); m++) {
		// m == 0 case
		for(int l = m; l <= static_cast<int>(l_max); l++) {
		    // do both m+ and m- at the same time (same Plm)
		    double p = plm(l, m, cos_theta[itheta]);

		    if(m & 1) {
			fft_work_array(m) += coeffs(coeff_idx) * p;
		    }
		    else {
			fft_work_array(m) += std::conj(coeffs(coeff_idx)) * p;
		    }
		    coeff_idx++;
		}
	    }
	    pocketfft::c2r(shape, stride, stride, axes, pocketfft::BACKWARD,
			   fft_work_array.data(), reinterpret_cast<double *>(fft_work_array.data()), 1.0);
	    values.row(itheta) = fft_work_array.real();
	}
	return values;
    }


    template<typename F>
    CMat values_on_grid_complex(F &f) {
	CMat values(m_ntheta, m_nphi);
	for(size_t i = 0; i < m_ntheta; i++) { 
	    for(size_t j = 0; j < m_nphi; j++) {
		values(i, j) = f(std::acos(cos_theta(i)), phi_pts(j));
	    }
	}
	return values;
    }

    template<typename F>
    Mat values_on_grid_real(F &f) {
	Mat values(m_ntheta, m_nphi);
	for(size_t i = 0; i < m_ntheta; i++) { 
	    for(size_t j = 0; j < m_nphi; j++) {
		values(i, j) = f(std::acos(cos_theta(i)), phi_pts(j));
	    }
	}
	return values;
    }


private:
    size_t l_max{0};
    size_t m_nphi{0}, m_ntheta{0};
    Vec phi_pts;
    Vec cos_theta;
    Vec weights;
    CVec fft_work_array;

    pocketfft::shape_t shape;
    pocketfft::shape_t axes{0};
    pocketfft::stride_t stride{sizeof(Complex)}; 
    AssocLegendreP plm;
};

}
