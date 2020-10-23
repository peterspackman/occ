#include "dft_grid.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include "logger.h"
#include "timings.h"

namespace tonto::dft {

const std::array<uint_fast16_t, 33> lebedev_grid_levels {
    1, 6, 14, 26, 38,
    50, 74, 86, 110, 146,
    170, 194, 230, 266, 302,
    350, 434, 590, 770, 974,
    1202, 1454, 1730, 2030, 2354,
    2702, 3074, 3470, 3890, 4334,
    4802, 5294, 5810
};

// Need to multiple by (1.0 / BOHR) to get the real RADII_BRAGG
const std::array<double, 131> bragg_radii = {
    0.35,                                     1.40,              // 1s
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,              // 2s2p
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,              // 3s3p
    2.20, 1.80,                                                  // 4s
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35,  // 3d
                1.30, 1.25, 1.15, 1.15, 1.15, 1.90,              // 4p
    2.35, 2.00,                                                  // 5s
    1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55,  // 4d
                1.55, 1.45, 1.45, 1.40, 1.40, 2.10,              // 5p
    2.60, 2.15,                                                  // 6s
    1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                    // La, Ce-Eu
    1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,              // Gd, Tb-Lu
          1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,  // 5d
                1.90, 1.80, 1.60, 1.90, 1.45, 2.10,              // 6p
    1.80, 2.15,                                                  // 7s
    1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
                1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75
};


tonto::IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular, size_t num_radial)
{
    std::array<int, 5> lebedev_level{4, 5, 5, 5, 4};
    tonto::IVec angular_grids(num_radial);
    if(max_angular < 50) {
        angular_grids.setConstant(max_angular);
    }
    else {
        size_t i;
        for(i = 6; i < lebedev_grid_levels.size(); i++) {
            if(lebedev_grid_levels[i] == max_angular) break;
        }
        lebedev_level[1] = 6;
        lebedev_level[2] = i - 1;
        lebedev_level[3] = i;
        lebedev_level[4] = i + 1;
    }

    std::array<double, 4> alphas;
    if(nuclear_charge <= 2) {
        alphas = {0.25, 0.5, 1.0, 4.5};
    }
    else if(nuclear_charge <= 10)
    {
        alphas = {0.16666667, 0.5, 0.9, 3.5};
    }
    else {
        alphas =  {0.1, 0.4, 0.8, 2.5};
    }
    constexpr double bohr{0.52917721092};
    double radius = bragg_radii[nuclear_charge - 1] / bohr;
    for(size_t i = 0; i < num_radial; i++)
    {
        double scale = angular_grids(i) / radius;
        size_t place;
        for(place = 0; place < 4; place++)
            if(scale <= alphas[place]) break;
        angular_grids(i) = lebedev_grid_levels[lebedev_level[place]];
    }
    return angular_grids;
}


DFTGrid::DFTGrid(
        const libint2::BasisSet &basis,
        const std::vector<libint2::Atom> &atoms) : m_atomic_numbers(atoms.size()),
    m_alpha_max(atoms.size()), m_l_max(atoms.size()), m_x(atoms.size()), m_y(atoms.size()),
    m_z(atoms.size())
{
    int i = 0;
    tonto::timing::start(tonto::timing::category::grid_init);
    const auto atom_map = basis.atom2shell(atoms);

    std::vector<std::vector<double>> min_alpha;
    for(const auto &atom : atoms) {
        m_atomic_numbers(i) = atom.atomic_number;
        m_x(i) = atom.x;
        m_y(i) = atom.y;
        m_z(i) = atom.z;
        std::vector<double> atom_min_alpha;
        double max_alpha = 0.0;
        int max_l = 0;
        for(const auto& shell_idx: atom_map[i])
        {
            const auto& shell = basis[shell_idx];
            int j = 0;
            for (const auto& contraction: shell.contr) {
                int l = contraction.l;
                max_l = std::max(max_l, l);
                j++;
            }

            for (int i = atom_min_alpha.size(); i < max_l + 1; i++) {
                atom_min_alpha.push_back(std::numeric_limits<double>::max());
            }
            j = 0;
            for (const auto& contraction: shell.contr) {
                int l = contraction.l;
                atom_min_alpha[l] = std::min(shell.alpha[j], atom_min_alpha[l]);
                j++;
            }

            for(const double alpha: shell.alpha) {
                max_alpha = std::max(alpha, max_alpha);
            }
        }
        min_alpha.push_back(atom_min_alpha);
        m_alpha_max(i) = max_alpha;
        m_l_max(i) = max_l;
        i++;
    }
    m_alpha_min = tonto::Mat::Zero(atoms.size(), m_l_max.maxCoeff() + 1);
    for(size_t i = 0; i < min_alpha.size(); i++) {
        for(size_t j = 0; j < min_alpha[i].size(); j++) {
            m_alpha_min(i, j) = min_alpha[i][j];
        }
    }
    tonto::timing::stop(tonto::timing::category::grid_init);
}

std::pair<Mat3N, Vec> DFTGrid::grid_points(size_t idx) const
{
    tonto::timing::start(tonto::timing::category::grid_points);
    assert(idx < m_atomic_numbers.size());
    context_t *ctx = numgrid_new_atom_grid(
        m_radial_precision,
        m_min_angular, m_max_angular,
        m_atomic_numbers(idx),
        m_alpha_max(idx),
        m_l_max(idx),
        m_alpha_min.row(idx).data()
    );
    tonto::log::debug("Context created");
    int num_points = numgrid_get_num_grid_points(ctx);

    Mat pts(num_points, 3);
    Vec weights(num_points);
    numgrid_get_grid(
        ctx,
        n_atoms(),
        idx,
        m_x.data(),
        m_y.data(),
        m_z.data(),
        m_atomic_numbers.data(),
        pts.col(0).data(),
        pts.col(1).data(),
        pts.col(2).data(),
        weights.data()
    );
    tonto::log::debug("Grid points assigned");

    numgrid_free_atom_grid(ctx);
    tonto::timing::stop(tonto::timing::category::grid_points);
    return {pts.transpose(), weights};
}

}
