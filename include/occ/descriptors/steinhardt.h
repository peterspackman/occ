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

private:
    struct Wigner3jCache {
	int l;
	int m1, m2, m3;
	double coeff;
    };


    size_t m_lmax;
    sht::SphericalHarmonics m_harmonics;
    std::vector<Wigner3jCache> m_wigner_coefficients;
};

}
