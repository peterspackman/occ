#include "dft.h"
#include "molecule.h"
#include "numgrid.h"
#include <fmt/core.h>
#include "xc.h"

namespace craso::dft {

DFTGrid::DFTGrid(const craso::chem::Molecule &mol) : m_atomic_numbers(mol.atomic_numbers()),
    m_alpha_max(mol.size()), m_l_max(mol.size()), m_x(mol.size()), m_y(mol.size()), m_z(mol.size())
{
    m_l_max = craso::IVec::Constant(m_l_max.rows(), m_l_max.cols(), 2);
    m_alpha_min = craso::Mat::Constant(mol.size(), m_l_max.maxCoeff(), 0.3);
    m_alpha_max = craso::Mat::Constant(m_alpha_max.rows(), m_alpha_max.cols(), 11720.0);
    const auto& pos = mol.positions();
    m_x = pos.row(0);
    m_y = pos.row(1);
    m_z = pos.row(2);
}


Mat4N DFTGrid::grid_points(size_t idx) const
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

    fmt::print("{} points in grid on atom {}\n", num_points, idx);
    Mat pts(num_points, 4);
    fmt::print("Points: ({} {})\n", pts.rows(), pts.cols());

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
    return pts.transpose();
}


}
