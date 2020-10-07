#include "dft_grid.h"
#include <libint2/basis.h>
#include <libint2/atom.h>

namespace tonto::dft {

DFTGrid::DFTGrid(
        const libint2::BasisSet &basis,
        const std::vector<libint2::Atom> &atoms) : m_atomic_numbers(atoms.size()),
    m_alpha_max(atoms.size()), m_l_max(atoms.size()), m_x(atoms.size()), m_y(atoms.size()),
    m_z(atoms.size())
{
    int i = 0;
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
}

MatN4 DFTGrid::grid_points(size_t idx) const
{
    assert(idx < m_atomic_numbers.size());
    context_t *ctx = numgrid_new_atom_grid(
        m_radial_precision,
        m_min_angular, m_max_angular,
        m_atomic_numbers(idx),
        m_alpha_max(idx),
        m_l_max(idx),
        m_alpha_min.row(idx).data()
    );
    int num_points = numgrid_get_num_grid_points(ctx);

    Mat pts(num_points, 4);
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
        pts.col(3).data()
    );

    numgrid_free_atom_grid(ctx);
    return pts;
}

}
