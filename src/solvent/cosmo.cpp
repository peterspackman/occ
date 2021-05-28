#include <occ/solvent/cosmo.h>
#include <occ/core/diis.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::solvent {

namespace cosmo {

occ::Vec solvation_radii(const occ::IVec &nums)
{
    // angstroms
    occ::Vec result(nums.rows());
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
            if(norm > 1e-3) coulomb(i, j) = 1.0 / norm;
            else coulomb(i, j) = 0.0;
            coulomb(j, i) = coulomb(i, j);

        }
    }

    coulomb.diagonal().setConstant(1.05 * sqrt(590));
    res.initial = surface_charge(charges);
    res.converged = Vec(res.initial.rows());

    res.converged = coulomb.llt().solve(res.initial);
    res.energy = - 0.5 * res.initial.dot(res.converged);
    return res;
}

}
