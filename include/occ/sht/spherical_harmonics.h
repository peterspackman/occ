#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/sht/legendre.h>

namespace occ::sht {

class SphericalHarmonics {

public:
    SphericalHarmonics(size_t lm, bool phase=true);
    CVec evaluate(Eigen::Ref<const Vec3> pos);
    void evaluate(Eigen::Ref<const Vec3> pos, Eigen::Ref<CVec>);

    CVec evaluate(double theta, double phi);
    void evaluate(double theta, double phi, Eigen::Ref<CVec>);

    inline auto nlm() const { return (m_lmax + 1) * (m_lmax + 1); }

private:
    bool m_phase{true};
    size_t m_lmax{0};
    AssocLegendreP m_plm_evaluator;
    Vec m_plm;
};

}
