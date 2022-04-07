#include <complex>
#include <cmath>
#include <vector>
#include <iostream>
#include <occ/3rdparty/pocketfft.h>
#include <occ/sht/sht.h>
#include <occ/core/logger.h>

namespace occ::sht {

std::pair<Vec, Vec> gauss_legendre_quadrature(int N) {
    Vec roots(N), weights(N);
    gauss_legendre_quadrature(roots, weights, N);
    return {roots, weights};
}

void gauss_legendre_quadrature(Vec &roots, Vec &weights, int N) {

    const double eps{2.3e-16};
    const long m = (N + 1) / 2;
    for(long i = 0; i < m; ++i) {
	double z1, deriv, point0, point1;
	// maximum Newton iteration count
	int iteration_count = 10;
	// initial guess
	double z = (1.0 - (N - 1) / (8.0 * N * N * N)) * 
		    std::cos((M_PI*(4 * i + 3)) / (4.0 * N + 2));
	do {
	    point1 = z;	// P_1
	    point0 = 1.0; // P_0
	    for(long l = 2; l <= N;++l) { 
		// recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}	(works ok up to l=100000)
		double point3 = point0;
		point0 = point1;
		// Legendre polynomial
		point1 = ((2 * l - 1) * z * point0 - (l - 1) * point3) / l;
	    }
	    // Approximate derivative of Legendre Polynomial
	    deriv = N * (point0 - z * point1);
	    z1 = z;
	    // Newton's method step
	    z -= point1 * (1.0 - z * z) / deriv;
	} while ((std::fabs(z - z1) > 
		  (z1 + z) * 0.5 * eps)
		&& (--iteration_count > 0));

	if(iteration_count == 0) occ::log::warn("Iterations exceeded when finding Gauss-Legendre roots");

	double s2 = 1.0 - z * z;
	// Build up the abscissas.
	roots(i) = z;
	roots(N - 1 - i) = -z;
	// Build up the weights.
	weights(i) = 2.0 * s2 / (deriv * deriv);
	weights(N - 1 - i) = weights(i);
    }
    // if n is even
    if(N & 1) {
	roots(N / 2) = 0.0; // exactly zero.
	weights(N / 2) = 1.0;
	double point0 = 1.0; // P_0
	for(long l = 2; l <= N; l += 2) {
	    // recurrence : l P_l = (2l-1) z P_{l-1} - (l-1) P_{l-2}	(works ok up to l=100000)
	    // The Legendre polynomial...
	    point0 *= (1.0 - l) / l;
	}
	// ... and its inverse derivative.
	double deriv = 1.0 / (N * point0);
	weights(N / 2) = 2.0 * deriv * deriv;
    }
    // as we started with initial guesses, we should check if the gauss points are actually unique and ordered.
    for (long i = m - 1; i > 0; i--) {
	if (roots(i) >= roots(i - 1)) occ::log::error("Invalid Gauss-Legendre points");
    }
}

AssocLegendreP::AssocLegendreP(size_t lm) : l_max(lm), m_a(lm + 1, lm + 1), m_b(lm + 1, lm + 1), m_cache(lm + 1, lm + 1) {
    for(size_t m = 0; m <= l_max; m++) {
	m_a(m, m) = amm(m);
	for(size_t l = m + 1; l <= l_max; l++) {
	    m_a(l, m) = alm(l, m);
	    m_b(l, m) = blm(l, m);
	}
    }
}

double AssocLegendreP::operator()(size_t l, size_t m, double x) const {
    if(m == l) {
	// abs(m) / 2
	return m_a(l, m) * std::pow(1 - x * x, 0.5 * m);
    }
    else if (m + 1 == l) {
	return m_a(l, m) * x * (*this)(m, m, x);
    }
    else {
	return (
	    m_a(l, m) * x * (*this)(l - 1, m, x) +
	    m_b(l, m) * (*this)(l - 2, m, x)
	);
    }
}

void AssocLegendreP::evaluate_batch(double x, Vec &result) const {
    size_t idx = 0;
    double tmp1 = 1 - x * x;
    double sqrt_tmp1 = std::sqrt(tmp1);
    for(size_t m = 0; m <= l_max; m++) {
	for(size_t l = m; l <= l_max; l++) {
	    if(l == m) {
		result(idx) = m_a(l, m) * std::pow(1 - x * x, 0.5 * m);
	    }
	    else if(l == (m + 1)) {
		result(idx) = m_a(l, m) * x * m_cache(l - 1, m);
	    }
	    else {
		result(idx) = m_a(l, m) * x * m_cache(l - 1, m) + m_b(l, m) * m_cache(l - 2, m);
	    }
	    m_cache(l, m) = result(idx);
	    idx++;
	}
    }
}

Vec AssocLegendreP::evaluate_batch(double x) const {
    Vec result((l_max + 1) * (l_max + 2) / 2);
    evaluate_batch(x, result);
    return result;
}


double AssocLegendreP::amm(size_t m) {
    const double pi4 = 4 * M_PI;
    double result {1.0};
    for(int k = 1; k <= m; k++) {
	result *= (2.0 * k + 1.0) / (2.0 * k);
    }
    return std::sqrt(result / pi4);
}

double AssocLegendreP::alm(size_t l, size_t m) {
    return std::sqrt((4.0 * l * l - 1) / (1.0 * l * l - m * m));
}

double AssocLegendreP::blm(size_t l, size_t m) {
    return - std::sqrt(
	(2.0 * l + 1) * ((l - 1.0)*(l - 1.0) - m * m) / 
	((2.0 * l - 3) * (1.0 * l * l - m * m)));
}

SHT::SHT(size_t lm) : l_max(lm), plm(lm), m_nphi(2 * lm + 1), shape({m_nphi}) {
    phi_pts = Vec(m_nphi);
    for(size_t i = 0; i < m_nphi; i++) {
	phi_pts(i) = (2 * M_PI * i) / m_nphi - M_PI;
    }
    m_ntheta = 1;
    while(m_ntheta  < m_nphi) {
	m_ntheta *= 2;
    }
    std::tie(cos_theta, weights) = gauss_legendre_quadrature(m_ntheta);
    fft_work_array = CVec(m_nphi);
}

}
