#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::solvent
{
using occ::Vec;
using occ::Mat3N;
using occ::Mat;

namespace cosmo {
occ::Vec solvation_radii(const occ::IVec&);
}

class COSMO
{

public:
    struct Result
    {
        occ::Vec initial;
        occ::Vec converged;
        double energy;
    };

    COSMO(double dielectric) : m_dielectric(dielectric) {}

    Result operator()(const Mat3N&, const Vec&, const Vec&) const;

    auto surface_charge(const Vec& charges) const
    {
        return charges.array() * (m_dielectric - 1) / (m_dielectric + m_x);
    }

    void set_max_iterations(size_t max_iter) { m_max_iterations = max_iter; }
    void set_x(double x) { m_x = x; }

private:
    double m_x{0.5};
    size_t m_diis_start{1};
    double m_dielectric;
    double m_diis_tolerance{1e-6};
    double m_convergence{1e-6};
    double m_initial_charge_scale_factor = 0.0694;
    size_t m_max_iterations{50};
};
}
