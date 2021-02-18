#include <tonto/solvent/cosmo.h>
#include <tonto/core/diis.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace tonto::solvent {

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

    std::vector<std::pair<Vec, Vec>> diis_vecs;
    double energy = 0.0;
    tonto::diis::DIIS<Vec> diis;


    for(size_t k = 1; k < m_max_iterations; k++)
    {
        vpot = (coulomb.array().colwise() * prev.array()).colwise().sum();
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
