#include <occ/solvent/surface.h>
#include <occ/dft/lebedev.h>

namespace occ::solvent::surface
{

occ::Mat3 principal_axes(const occ::Mat3N &positions)
{
    Eigen::JacobiSVD<occ::Mat> svd(positions, Eigen::ComputeThinU);
    return svd.matrixU();
}

Surface solvent_surface(const occ::Vec &radii, const occ::IVec &atomic_numbers, const occ::Mat3N &positions)
{
    const size_t N = atomic_numbers.rows();
    const size_t npts = 110;
    const double delta = 0.1889725127952816; // 0.1 Angstrom
    Surface surface;
    surface.vertices = occ::Mat3N(3, N * npts);
    surface.areas = occ::Vec(N * npts);
    surface.atom_index = occ::IVec(N * npts);
    auto grid = occ::grid::lebedev(npts);
    occ::Mat3N lebedev_points = grid.leftCols(3).transpose();
    occ::Vec lebedev_weights = grid.col(3);
    occ::Vec3 centroid = positions.rowwise().mean();
    occ::Mat3N centered = positions.colwise() - centroid;
    auto axes = principal_axes(centered);
    centered = axes.transpose() * centered;

    for(size_t i = 0; i < N; i++)
    {
        double r = radii(i) + delta;
        surface.areas.segment(npts * i, npts) = lebedev_weights * 4 * M_PI * radii(i) * radii(i);
        auto vblock = surface.vertices.block(0, npts * i, 3, npts);
        vblock = lebedev_points * r;
        vblock.colwise() += centered.col(i);
        surface.atom_index.segment(npts * i, npts).array() = i;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask(N * npts);
    mask.setConstant(true);
    
    size_t num_valid_points = surface.vertices.cols();

    for(size_t i = 0; i < N; i++)
    {
        occ::Vec3 q = centered.col(i);
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

    occ::Mat3N remaining_points(3, num_valid_points);
    occ::Vec remaining_weights(num_valid_points);
    occ::IVec remaining_atom_index(num_valid_points);
    size_t j = 0;
    for(size_t i = 0; i < mask.rows(); i++)
    {
        if(mask(i)) {
            size_t atom_idx = surface.atom_index(i);
            occ::Vec3 v = surface.vertices.col(i);
            occ::Vec3 shift = (v - centered.col(atom_idx));
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
    std::swap(remaining_weights, surface.areas);
    std::swap(remaining_atom_index, surface.atom_index);
    surface.vertices = (axes * remaining_points).colwise() + centroid;
    return surface;
}

}
