#include <occ/solvent/surface.h>
#include <occ/dft/lebedev.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::solvent::surface
{

occ::Mat3 principal_axes(const occ::Mat3N &positions)
{
    if(positions.cols() == 1) return occ::Mat3::Identity();
    Eigen::JacobiSVD<occ::Mat> svd(positions, Eigen::ComputeThinU);
    occ::Mat3 result = svd.matrixU();
    occ::Vec proportions = svd.singularValues();
    if(proportions.rows() < 3) {
        result.col(2) = result.col(0).cross(result.col(1));
    }
    return result;
}

Surface solvent_surface(const occ::Vec &radii, const occ::IVec &atomic_numbers, const occ::Mat3N &positions)
{
    const size_t N = atomic_numbers.rows();
    const size_t npts = 302;
    const size_t npts_H = 302;
    const double solvent_radius = 1.0;
    const double delta = 0.2 * solvent_radius;
    Surface surface;
    size_t num_h{0}, num_other{0};
    for(size_t i = 0; i < N; i++)
    {
        if (atomic_numbers(i) == 1) num_h++;
        else num_other++;
    }
    size_t pt_count{npts * num_other + num_h * npts_H};
    auto tmp_vertices = occ::Mat3N(3, pt_count);
    auto tmp_areas = occ::Vec(pt_count);
    auto tmp_atom_index = occ::IVec(pt_count);
    auto grid = occ::grid::lebedev(npts);
    auto hgrid = occ::grid::lebedev(npts_H);
    occ::Mat3N lebedev_points = grid.leftCols(3).transpose();
    occ::Mat3N h_lebedev_points = hgrid.leftCols(3).transpose();
    occ::Vec lebedev_weights = grid.col(3);
    occ::Vec h_lebedev_weights = hgrid.col(3);
    occ::Vec3 centroid = positions.rowwise().mean();
    occ::Mat3N centered = positions.colwise() - centroid;
    auto axes = principal_axes(centered);
    centered = axes.transpose() * centered;

    occ::Vec ri = radii.array() + solvent_radius;

    size_t num_valid_points{0};
    for(size_t i = 0; i < N; i++)
    {
        double r = ri(i);
        if (atomic_numbers(i) == 1)
        {
            tmp_areas.segment(num_valid_points, npts_H) = 
                h_lebedev_weights * 4 * M_PI * r * r;
            auto vblock = tmp_vertices.block(0, num_valid_points, 3, npts_H);
            vblock = h_lebedev_points * r;
            vblock.colwise() += centered.col(i);
            tmp_atom_index.segment(num_valid_points, npts_H).array() = i;
            num_valid_points += npts_H;
        }
        else
        {
            tmp_areas.segment(num_valid_points, npts) = 
                lebedev_weights * 4 * M_PI * r * r;
            auto vblock = tmp_vertices.block(0, num_valid_points, 3, npts);
            vblock = lebedev_points * r;
            vblock.colwise() += centered.col(i);
            tmp_atom_index.segment(num_valid_points, npts).array() = i;
            num_valid_points += npts;
        }
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask(num_valid_points);
    mask.setConstant(true);
    

    for(size_t i = 0; i < N; i++)
    {
        occ::Vec3 q = centered.col(i);
        for(size_t j = 0; j < tmp_vertices.cols(); j++)
        {
            if(!mask(j) || tmp_atom_index(j) == i) continue;
            double r = (q - tmp_vertices.col(j)).norm();
            if(r < ri(i)) {
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
            size_t atom_idx = tmp_atom_index(i);
            occ::Vec3 v = tmp_vertices.col(i);
            occ::Vec3 shift = (v - centered.col(atom_idx));
            shift.normalize();
            shift.array() *= delta;
            // shift the position back by delta
            v -= shift;
            remaining_points.col(j) = v;
            remaining_weights(j) = tmp_areas(i);
            remaining_atom_index(j) = atom_idx;
            j++;
        }
    }
    surface.areas = remaining_weights;
    surface.atom_index = remaining_atom_index;
    surface.vertices = (axes * remaining_points).colwise() + centroid;
    return surface;
}

}
