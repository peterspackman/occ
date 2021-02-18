#pragma once
#include <tonto/core/linear_algebra.h>

namespace tonto::solvent
{
using tonto::Vec;
using tonto::Mat3N;
using tonto::Mat;

class COSMO
{

public:
    struct Result
    {
        tonto::Vec initial;
        tonto::Vec converged;
        double energy;
    };

    COSMO(double dielectric) : m_dielectric(dielectric) {}

    Result operator()(const Mat3N&, const Vec&, const Vec&) const;

    auto surface_charge(const Vec& charges) const
    {
        return charges.array() * (m_dielectric - 1) / (m_dielectric + m_x);
    }


private:
    const double m_x{0.5};
    const size_t m_diis_start{1};
    const double m_dielectric;
    const double m_diis_tolerance{1e-6};
    const double m_convergence{1e-6};
    const double m_initial_charge_scale_factor = 0.0694;
    const size_t m_max_iterations{50};
};
}
