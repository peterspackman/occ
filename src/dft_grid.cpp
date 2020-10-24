#include "dft_grid.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include "logger.h"
#include "timings.h"
#include "lebedev.h"

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

tonto::Vec becke_partition(const tonto::Vec &w)
{
    tonto::Vec result = w;
    for(size_t i = 0; i < 3; i++)
    {
        result(i) = (3 - result(i) * result(i)) * result(i) * 0.5;
    }
    return result;
}

tonto::Vec stratmann_scuseria_partition(const tonto::Vec &w)
{
    tonto::Vec result;
    constexpr double a = 0.64;
    for(size_t i = 0; i < w.size(); i++)
    {
        double ma = w(i) / a;
        double ma2 = ma * ma;
        double det = ma/16 * (35 + ma2 * (-35 + ma2 * (21 - 5 * ma2)));
        result(i) = (det <= a) ? -1 : 1;
    }
    return result;
}

tonto::Mat interatomic_distances(const std::vector<libint2::Atom> & atoms)
{
    size_t natoms = atoms.size();
    tonto::Mat dists(natoms, natoms);
    for (size_t i = 0; i < natoms; i++)
    {
        dists(i, i) = 0;
        for(size_t j = i + 1; j < natoms; j++) 
        {
            double dx = atoms[i].x - atoms[j].x;
            double dy = atoms[i].y - atoms[j].y;
            double dz = atoms[i].z - atoms[j].z;
            dists(i, j) = sqrt(dx*dx + dy*dy + dz*dz);
            dists(j, i) = dists(i, j);
        }
    }
    return dists;
}

AtomGrid partitioned_atom_grid(size_t atom_idx, const std::vector<libint2::Atom>& atoms, const AtomGrid &atom_type_grid)
{
    size_t natoms = atoms.size();
    const tonto::Mat dists = interatomic_distances(atoms);
    tonto::Vec3 center(atoms[atom_idx].x, atoms[atom_idx].y, atoms[atom_idx].z);
    AtomGrid grid = atom_type_grid;
    grid.points.colwise() += center;
    tonto::Mat grid_dists(natoms, grid.num_points());
    for(size_t i = 0; i < natoms; i++) 
    {
        tonto::Vec3 xyz(atoms[i].x, atoms[i].y, atoms[i].z);
        grid_dists.row(i) = (grid.points.colwise() - xyz).colwise().norm();
    }
    tonto::Mat becke_weights = tonto::Mat::Ones(natoms, grid.num_points());
    for(size_t i = 0; i < natoms; i++)
    {
        for(size_t j = 0; j < i; j++)
        {
            tonto::Vec w = (1 / dists(i, j)) * (grid_dists.row(i).array()  - grid_dists.row(j).array());
            w = becke_partition(w);
            becke_weights.row(i).array() *= 0.5 * (1 - w.array());
            becke_weights.row(j).array() *= 0.5 * (1 + w.array());
        }
    }
    grid.weights.array() *= becke_weights.row(atom_idx).array() * (1 / becke_weights.array().rowwise().sum());

    return grid;
}

tonto::IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular, size_t num_radial, const tonto::Vec& radii)
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
        double scale = radii(i) / radius;
        size_t place;
        for(place = 0; place < 4; place++)
            if(scale <= alphas[place]) break;
        angular_grids(i) = lebedev_grid_levels[lebedev_level[place]];
    }
    return angular_grids;
}

// Mura-Knowles [JCP 104, 9848 (1996) - doi:10.1063/1.471749] log3 quadrature
RadialGrid generate_mura_knowles_radial_grid(size_t num_points, size_t charge)
{
    RadialGrid result(num_points);
    double far = 5.2;
    switch(charge) {
    case 3:
    case 4:
    case 11:
    case 12:
    case 19:
    case 20:
        far = 7;
        break;
    }
    for(size_t i = 0; i < num_points; i++)
    {
        double x = (i + 0.5) / num_points;
        double x2 = x * x;
        double x3 = x2 * x;
        result.points(i) = - far * std::log(1 - x3);
        result.weights(i) = far * x2 / ((1 - x3) * num_points);
    }
    return result;
}

// Becke [JCP 88, 2547 (1988) - doi:10.1063/1.454033] quadrature
RadialGrid generate_becke_radial_grid(size_t num_points, double rm)
{
    double pi_npt1 = M_PI / (num_points + 1);
    double rm3 = rm * rm * rm;
    RadialGrid result(num_points);
    for(size_t i = 1; i <= num_points; i++)
    {
        double radx = cos(i * pi_npt1);
        double radr = rm * (1 + radx) / (1 - radx);
        double tmp0 = pow(1 + radx, 2.5);
        double tmp1 = pow(1 - radx, 3.5);
        double radw = (2 * pi_npt1) * rm3 * tmp0 / tmp1;
        result.points(num_points - i) = radr;
        result.weights(num_points - i) = radw;
    }
    return result;
}

// Treutler-Alrichs [JCP 102, 346 (1995) - doi:10.1063/1.469408] M4 quadrature
RadialGrid generate_treutler_alrichs_radial_grid(size_t num_points)
{
    RadialGrid result(num_points);
    double step = M_PI / (num_points + 1);
    double ln2 = 1 / std::log(2);
    for(size_t i = 1; i <= num_points; i++)
    {
        double x = cos(i * step);
        double tmp1 = ln2 * pow((1 + x), 0.6);
        double tmp2 = std::log((1 - x) / 2);
        result.points(num_points - i) = - tmp1 * tmp2;
        result.weights(num_points - i) =
            step * sin(i * step) * tmp1 * (-0.6 / (1 + x) * tmp2  + 1 / (1 - x));
    }
    return result;
}

AtomGrid generate_atom_grid(size_t atomic_number, size_t max_angular_points, size_t radial_points)
{
    const double rm = 1.0;
    AtomGrid result(radial_points * max_angular_points);

    size_t num_points = 0;
    size_t n_radial = radial_points;
    if(atomic_number <= 10) n_radial = 50;
    if(atomic_number <= 2) n_radial = 35;
    RadialGrid radial = generate_becke_radial_grid(n_radial, rm);
    tonto::IVec n_angular = prune_nwchem_scheme(atomic_number, max_angular_points, n_radial, radial.points);
    for(size_t i = 0; i < n_radial; i++)
    {
        auto lebedev = tonto::grid::lebedev(n_angular(i));
        result.points.block(0, num_points, 3, lebedev.rows()) = lebedev.leftCols(3).transpose() * radial.points(i);
        result.weights.block(num_points, 0, lebedev.rows(), 1) = lebedev.col(3) * 4 * M_PI * radial.weights(i);
        num_points += lebedev.rows();
    }
    result.points.conservativeResize(3, num_points);
    result.weights.conservativeResize(num_points);
    return result;
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
