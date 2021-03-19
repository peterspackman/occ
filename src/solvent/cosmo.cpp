#include <tonto/solvent/cosmo.h>
#include <tonto/core/diis.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace tonto::solvent {

namespace cosmo {

tonto::Vec solvation_radii(const tonto::IVec &nums)
{
    // angstroms
    tonto::Vec result(nums.rows());
    static const double radii[17] = {
        1.300, 1.638, 1.404, 1.053, 2.0475, 2.00,  
        1.830, 1.720, 1.720, 1.8018, 1.755, 1.638,  
        1.404, 2.457, 2.106, 2.160, 2.05
    };

    for(size_t i = 0; i < nums.rows(); i++)
    {
        int n = nums(i);
        double r = 2.223;
        if(n <= 17 && n > 0) r = radii[n - 1];
        result(i) = r;
    }
    return result;
}

}

COSMO::Result COSMO::operator()(const Mat3N &positions, const Vec &areas, const Vec &charges) const
{
    COSMO::Result res;
    Mat coulomb(positions.cols(), positions.cols());
    for(size_t i = 0; i < positions.cols(); i++)
    {
        for(size_t j = i + 1; j < positions.cols(); j++)
        {
            double norm = (positions.col(i) - positions.col(j)).norm();
            if(norm != 0.0) coulomb(i, j) = 1.0 / norm;
            else coulomb(i, j) = 0.0;
            coulomb(j, i) = coulomb(i, j);

        }
    }

    coulomb.diagonal().setConstant(0.0);
    Vec d0 = areas.array().sqrt() / 3.8;
    res.initial = surface_charge(charges);
    res.converged = Vec(res.initial.rows());

    Vec prev = m_initial_charge_scale_factor * res.initial.array() * d0.array();
    Vec vpot(coulomb.rows());
    Vec dq(res.initial.rows());

    double energy = 0.0;
    tonto::diis::DIIS<Vec> diis(2, 12);


    for(size_t k = 1; k < m_max_iterations; k++)
    {
        vpot.array() = (coulomb.array().colwise() * prev.array()).colwise().sum();
        res.converged = (res.initial.array() - vpot.array()) * d0.array();
        dq.array() = res.converged.array() - prev.array();
        diis.extrapolate(res.converged, dq);
        double rms_error = sqrt(dq.dot(dq) / dq.rows());

        res.energy = -0.5 * res.initial.dot(res.converged);
        fmt::print("{:3d} {:14.8f} {:9.5f} {:16.9f}\n", k, res.energy, res.converged.sum(), rms_error);
        if(rms_error < m_convergence) break;
        prev.array() = res.converged.array();
    }
    return res;
}

}
