#include <occ/descriptors/steinhardt.h>
#include <occ/sht/wigner3j.h>

using occ::sht::wigner3j_single;

namespace occ::descriptors {
Steinhardt::Steinhardt(size_t lmax) : m_lmax(lmax), m_harmonics(lmax) {}

CVec Steinhardt::compute_qlm(Eigen::Ref<const Mat3N> positions) {
    size_t num_particles = positions.cols();

    CVec ylm = CVec(m_harmonics.nlm());
    CVec qlm = CVec::Zero(m_harmonics.nlm());

    for (size_t i = 0; i < num_particles; ++i) {
        Vec3 r = positions.col(i);
	Vec3 pos = r.normalized();
        m_harmonics.evaluate(pos, ylm);

	int idx = 0;
        for (int l = 0; l <= m_lmax; l++) {
            for (int m = -l; m <= l; m++) {
                qlm(idx) += ylm(idx);
		idx++;
            }
        }
    }

    qlm /= num_particles;
    return qlm;
}

Vec Steinhardt::compute_q(Eigen::Ref<const Mat3N> positions) {
    Vec q = Vec::Zero(m_lmax + 1);
    CVec qlm = compute_qlm(positions);
    int idx = 0;
    for (int l = 0; l <= m_lmax; l++) {
	double ql = 0.0;
	for (int m = -l; m <= l; m++) {
	    ql += std::norm(qlm(idx));
	    idx++;
	}
	ql = std::sqrt((4.0 * M_PI / (2.0 * l + 1.0)) * ql);
	q(l) += ql;
    }

    return q;
}

void Steinhardt::precompute_wigner3j_coefficients() {
    m_wigner_coefficients.clear();
    for (int l = 0; l <= m_lmax; l++) {
        for (int m1 = -l; m1 <= l; m1++) {
            for (int m2 = -l; m2 <= l; m2++) {
                for (int m3 = -l; m3 <= l; m3++) {
                    if (m1 + m2 + m3 == 0) {
                        double w3j = wigner3j_single(l, l, l, m1, m2, m3);
			if(w3j != 0.0)
			    m_wigner_coefficients.push_back({
				l, m1, m2, m3, w3j
			    });
                    }
                }
            }
        }
    }
}

Vec Steinhardt::compute_w(Eigen::Ref<const Mat3N> positions) {
    if(m_wigner_coefficients.size() == 0) {
	precompute_wigner3j_coefficients();
    }

    Vec w = Vec::Zero(m_lmax + 1);
    CVec qlm = compute_qlm(positions);


    for(const auto [l, m1, m2, m3, w3j] : m_wigner_coefficients) {
        int idx1 = l * l + l + m1;
        int idx2 = l * l + l + m2;
        int idx3 = l * l + l + m3;
        w(l) += std::abs(qlm(idx1) * qlm(idx2) * qlm(idx3) * w3j);
    }

    return w / std::pow(qlm.squaredNorm(), 1.5);
}

}
