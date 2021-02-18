#include <tonto/solvent/cosmo.h>
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
            coulomb(i, j) = 1.0 / (positions.col(i) - positions.col(j)).norm();
            coulomb(j, i) = coulomb(i, j);

        }
    }
    coulomb.diagonal().setConstant(0.0);
    Vec d0 = areas.array().sqrt() / 3.8;
    res.initial = surface_charge(charges);
    res.converged = Vec(res.initial.rows());

    Vec prev = m_initial_charge_scale_factor * res.initial.array() * d0.array();
    Vec vpot(coulomb.rows());

    std::vector<std::pair<Vec, Vec>> diis_vecs;
    double energy = 0.0;


    for(size_t k = 1; k < m_max_iterations; k++)
    {
        vpot = (coulomb.array().colwise() * prev.array()).colwise().sum();
        res.converged = (res.initial.array() - vpot.array()) * d0.array();
        Vec dq = res.converged.array() - prev.array();

        if (k > m_diis_start)
        {
            diis_vecs.push_back({res.converged, dq});
        }

        size_t num_diis = diis_vecs.size();

        if (num_diis > 1)
        {
            Vec rhs = Vec::Zero(num_diis + 1);
            rhs(num_diis) = -1;
            Mat B = Mat::Zero(num_diis + 1, num_diis + 1);
            B.col(num_diis).setConstant(-1);
            B.row(num_diis).setConstant(-1);
            B(num_diis, num_diis) = 0;
            for(size_t i = 0; i < diis_vecs.size(); i++)
            {
                for(size_t j = i; j < diis_vecs.size(); j++)
                {
                    B(i, j) = diis_vecs[i].second.dot(diis_vecs[j].second);
                    if(i != j) B(j, i) = B(i, j);
                }
            }
            Vec c = B.ldlt().solve(rhs).topRows(num_diis);

            res.converged.setZero();
            std::vector<std::pair<Vec, Vec>> new_diis_vecs;
            for(size_t i = 0; i < num_diis; i++)
            {
                res.converged.array() += c(i) * diis_vecs[i].first.array();
                if(c(i) >= m_diis_tolerance) new_diis_vecs.push_back(diis_vecs[i]);
            }

            std::swap(new_diis_vecs, diis_vecs);

        }
        double rms_error = sqrt(dq.dot(dq) / dq.rows());
        res.energy = -0.5 * res.initial.dot(res.converged);
        fmt::print("{:3d} {:14.8f} {:9.5f} {:16.9f}\n", k, res.energy, res.converged.sum(), rms_error);
        if(rms_error < m_convergence) break;
        prev.array() = res.converged.array();
    }
    return res;
}

}
