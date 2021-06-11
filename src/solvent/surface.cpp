#include <occ/solvent/surface.h>
#include <occ/dft/lebedev.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/core/units.h>
#include <cstring>
#include <occ/solvent/smd.h>

namespace occ::solvent::surface
{

Mat3 principal_axes(const Mat3N &positions)
{
    if(positions.cols() == 1) return Mat3::Identity();
    Eigen::JacobiSVD<Mat> svd(positions, Eigen::ComputeThinU);
    Mat3 result = svd.matrixU();
    Vec proportions = svd.singularValues();
    if(proportions.rows() < 3) {
        result.col(2) = result.col(0).cross(result.col(1));
    }
    return result;
}

Surface solvent_surface(const Vec &radii, const IVec &atomic_numbers, const Mat3N &positions, double solvent_radius_angs)
{
    const size_t N = atomic_numbers.rows();
    const double solvent_radius = solvent_radius_angs * occ::units::ANGSTROM_TO_BOHR;
    Surface surface;
    Mat tmp_vertices;
    Vec tmp_areas;
    IVec tmp_atom_index;
    Vec3 centroid = positions.rowwise().mean();
    Mat3N centered = positions.colwise() - centroid;
    auto axes = principal_axes(centered);
    centered = axes.transpose() * centered;

    const double points_per_bohr2{1.5};
    Vec ri = radii.array();
    const std::array<int, 6> point_levels{302, 350, 434, 590, 770, 974};

    size_t num_valid_points{0};
    for(size_t i = 0; i < N; i++)
    {
        double rs = ri(i);
        double r = rs + solvent_radius; 
        double surface_area = 4 * M_PI * rs * rs;
        double grid_size_d = surface_area * points_per_bohr2;
        int npts = point_levels[point_levels.size() - 1];
        for(int j = point_levels.size() - 1; j >= 0; j--)
        {
            if(grid_size_d > point_levels[j]) break;
            npts = point_levels[j];
        }
        tmp_vertices.conservativeResize(3, num_valid_points + npts);
        tmp_areas.conservativeResize(num_valid_points + npts);
        tmp_atom_index.conservativeResize(num_valid_points + npts);
        auto grid = occ::grid::lebedev(npts);
        tmp_areas.segment(num_valid_points, npts) = grid.col(3) * surface_area;
        auto vblock = tmp_vertices.block(0, num_valid_points, 3, npts);
        vblock = grid.block(0, 0, npts, 3).transpose() * r;
        vblock.colwise() += centered.col(i);
        tmp_atom_index.segment(num_valid_points, npts).array() = i;
        num_valid_points += npts;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, 1> mask(num_valid_points);
    mask.setConstant(true);
    

    for(size_t i = 0; i < N; i++)
    {
        Vec3 q = centered.col(i);
        for(size_t j = 0; j < tmp_vertices.cols(); j++)
        {
            if(!mask(j) || tmp_atom_index(j) == i) continue;
            double r = (q - tmp_vertices.col(j)).norm();
            if(r < (ri(i) + solvent_radius)) {
                num_valid_points--;
                mask(j) = false;
            }
        }
    }

    Mat3N remaining_points(3, num_valid_points);
    Vec remaining_weights(num_valid_points);
    IVec remaining_atom_index(num_valid_points);
    size_t j = 0;
    for(size_t i = 0; i < mask.rows(); i++)
    {
        if(mask(i)) {
            size_t atom_idx = tmp_atom_index(i);
            Vec3 v = tmp_vertices.col(i);
            Vec3 shift = (v - centered.col(atom_idx));
            shift.normalize();
            shift.array() *= solvent_radius;
            // shift the position back by solvent radius
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


IVec nearest_atom_index(const Mat3N &atom_positions, const Mat3N &element_centers)
{
    IVec result(element_centers.cols());
    Vec3 c, atom;
    for(int i = 0; i < element_centers.cols(); i++)
    {
        Eigen::Index idx{0};
        double dist = (atom_positions.colwise() - element_centers.col(i)).colwise().squaredNorm().minCoeff(&idx);
        result(i) = idx;
    }
    return result;
}

}
