#include <tonto/solvent/surface.h>
#include <tonto/dft/lebedev.h>
#include <fmt/core.h>

namespace tonto::solvent::surface
{

Surface solvent_surface(const tonto::Vec &radii, const tonto::IVec &atomic_numbers, const tonto::Mat3N &positions)
{
    const size_t N = atomic_numbers.rows();
    const size_t npts = 302;
    const double delta = 0.1889725127952816; // 0.1 Angstrom
    Surface surface;
    surface.vertices = tonto::Mat3N(3, N * npts);
    surface.areas = tonto::Vec(N * npts);
    surface.atom_index = tonto::IVec(N * npts);
    auto grid = tonto::grid::lebedev(npts);
    tonto::Mat3N lebedev_points = grid.leftCols(3).transpose();
    tonto::Vec lebedev_weights = grid.col(3);
    for(size_t i = 0; i < N; i++)
    {
        double r = radii(i) + delta;
        surface.areas.segment(npts * i, npts) = lebedev_weights * 4 * M_PI * r * r;
        auto vblock = surface.vertices.block(0, npts * i, 3, npts);
        vblock = lebedev_points * r;
        vblock.colwise() += positions.col(i);
        surface.atom_index.segment(npts * i, npts).array() = i;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask(N * npts);
    mask.setConstant(true);
    
    size_t num_valid_points = surface.vertices.cols();

    for(size_t i = 0; i < N; i++)
    {
        tonto::Vec3 q = positions.col(i);
        double radius = radii(i) + delta;
        for(size_t j = 0; j < surface.vertices.cols(); j++)
        {
            if(!mask(j) || surface.atom_index(j) == i) continue;
            double r = (q - surface.vertices.col(j)).norm();
            if(r < radius) {
                num_valid_points--;
                mask(j) = false;
            }
        }
    }

    tonto::Mat3N remaining_points(3, num_valid_points);
    tonto::Vec remaining_weights(num_valid_points);
    tonto::IVec remaining_atom_index(num_valid_points);
    size_t j = 0;
    for(size_t i = 0; i < mask.rows(); i++)
    {
        if(mask(i)) {
            size_t atom_idx = surface.atom_index(i);
            tonto::Vec3 v = surface.vertices.col(i);
            tonto::Vec3 shift = (v - positions.col(atom_idx));
            shift.normalize();
            shift.array() *= delta;
            // shift the position back by delta
            v -= shift;
            remaining_points.col(j) = v;
            remaining_weights(j) = surface.areas(i);
            remaining_atom_index(j) = atom_idx;
            j++;
        }
    }
    std::swap(remaining_points, surface.vertices);
    std::swap(remaining_weights, surface.areas);
    std::swap(remaining_atom_index, surface.atom_index);
    return surface;
}

}
