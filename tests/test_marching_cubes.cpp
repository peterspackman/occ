#include "linear_algebra.h"
#include "sparsevoxelgrid.h"
#include "marchingcubes.h"
#include "catch.hpp"
#include <fmt/ostream.h>
#include "timings.h"

TEST_CASE("Sphere") {
    using tonto::grid::sparse::voxel_grid;
    using tonto::geom::marching_cubes;

    std::function<double(const Eigen::RowVector3d&)> sphere = [](const Eigen::RowVector3d& pt) -> double {
        return 1.0 - pt.squaredNorm();
    };

    Eigen::RowVector3d p0(0., 0., 1.);
    // CS will hold one scalar value at each cube vertex corresponding
    // the value of the implicit at that vertex
    tonto::Vec CS;

    // CV will hold the positions of the corners of the sparse voxel grid
    tonto::Mat CV;

    // CI is a #cubes x 8 matrix of indices where each row contains the
    // indices into CV of the 8 corners of a cube
    Eigen::MatrixXi CI;

    const double eps = 0.1;
    // Construct the voxel grid, populating CS, CV, and CI
    tonto::timing::StopWatch<1> sw;
    sw.start(0);
    voxel_grid(p0, sphere, eps, CS, CV, CI);
    sw.stop(0);
    fmt::print("Voxel grid took: {}\n", sw.read(0));
    fmt::print("Scalars: {}\n", CS.size());
    fmt::print("Positions: {} {}\n", CV.rows(), CV.cols());
    fmt::print("Cube indices: {} {}\n", CI.rows(), CI.cols());

    Eigen::MatrixXi faces;
    Eigen::MatrixXd vertices;

    sw.clear_all();
    sw.start(0);
    marching_cubes(CS, CV, CI, vertices, faces);
    sw.stop(0);
    fmt::print("Marching cubes took {}\n", sw.read(0));
    fmt::print("Faces: {} {}\n", faces.rows(), faces.cols());
    fmt::print("Vertices: {} {}\n", vertices.rows(), vertices.cols());
}
