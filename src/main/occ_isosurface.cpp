#include <chrono>
#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/kdtree.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/numpy.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/io/xyz.h>
#include <occ/main/isosurface.h>
#include <occ/main/occ_isosurface.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Element;
using occ::core::Interpolator1D;
using occ::core::Molecule;
using occ::main::PromoleculeDensityFunctor;
using occ::main::StockholderWeightFunctor;

struct IsosurfaceMesh {
    IsosurfaceMesh() {}
    IsosurfaceMesh(size_t num_vertices, size_t num_faces)
        : vertices(3, num_vertices), faces(3, num_faces),
          normals(3, num_vertices) {}
    Eigen::Matrix3Xf vertices;
    Eigen::Matrix3Xi faces;
    Eigen::Matrix3Xf normals;
};

struct VertexProperties {
    VertexProperties() {}
    VertexProperties(int size)
        : de(size), di(size), de_idx(size), di_idx(size), de_norm(size),
          di_norm(size), de_norm_idx(size), di_norm_idx(size), dnorm(size) {}
    Eigen::VectorXf de;
    Eigen::VectorXf di;
    Eigen::VectorXi de_idx;
    Eigen::VectorXi di_idx;
    Eigen::VectorXf de_norm;
    Eigen::VectorXf di_norm;
    Eigen::VectorXi de_norm_idx;
    Eigen::VectorXi di_norm_idx;
    Eigen::VectorXf dnorm;
};

void write_ply_file(const std::string &filename,
                    const Eigen::Matrix3Xf &vertices,
                    const Eigen::Matrix3Xi &faces) {
    auto file = fmt::output_file(filename);
    file.print("ply\n");
    file.print("format ascii 1.0\n");
    file.print("comment exported from CrystalExplorer\n");
    file.print("element vertex {}\n", vertices.size() / 3);
    file.print("property float x\n");
    file.print("property float y\n");
    file.print("property float z\n");
    file.print("element face {}\n", faces.size() / 3);
    file.print("property list uchar int vertex_index\n");
    file.print("end_header\n");
    for (size_t idx = 0; idx < vertices.cols(); idx++) {
        file.print("{} {} {}\n", vertices(0, idx), vertices(1, idx),
                   vertices(2, idx));
    }
    for (size_t idx = 0; idx < faces.cols(); idx++) {
        file.print("3 {} {} {}\n", faces(0, idx), faces(1, idx), faces(2, idx));
    }
}

void write_obj_file(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties) {
    auto file = fmt::output_file(filename);
    file.print("# vertices\n");
    for (size_t idx = 0; idx < mesh.vertices.cols(); idx++) {
        file.print("v {} {} {}\n", mesh.vertices(0, idx), mesh.vertices(1, idx),
                   mesh.vertices(2, idx));
    }
    file.print("# vertex normals\n");
    for (size_t idx = 0; idx < mesh.vertices.cols(); idx++) {
        file.print("vn {} {} {}\n", mesh.normals(0, idx), mesh.normals(1, idx),
                   mesh.normals(2, idx));
    }
    file.print("# faces\n");
    for (size_t idx = 0; idx < mesh.faces.cols(); idx++) {
        int f1 = mesh.faces(0, idx) + 1;
        int f2 = mesh.faces(1, idx) + 1;
        int f3 = mesh.faces(2, idx) + 1;
        file.print("f {}/{} {}/{} {}/{}\n", f1, f1, f2, f2, f3, f3);
    }
    file.print("# dnorm di\n");
    for (size_t idx = 0; idx < properties.dnorm.rows(); idx++) {
        file.print("vt {} {} {}\n", properties.dnorm(idx), properties.de(idx),
                   properties.de(idx));
    }
}

template <typename F>
IsosurfaceMesh as_mesh(const F &b, const std::vector<float> &vertices,
                       const std::vector<uint32_t> &indices,
                       const std::vector<float> &normals) {

    IsosurfaceMesh result(vertices.size() / 3, indices.size() / 3);

    float length = b.side_length();
    const auto &origin = b.origin();
    for (size_t i = 0; i < vertices.size(); i += 3) {
        result.vertices(0, i / 3) = vertices[i] * length + origin(0);
        result.vertices(1, i / 3) = vertices[i + 1] * length + origin(1);
        result.vertices(2, i / 3) = vertices[i + 2] * length + origin(2);

        Eigen::Vector3f normal(normals[i], normals[i + 1], normals[i + 2]);
        result.normals.col(i / 3) = normal.normalized();
    }

    result.vertices.array() *= occ::units::BOHR_TO_ANGSTROM;

    for (size_t i = 0; i < indices.size(); i += 3) {
        result.faces(0, i / 3) = indices[i];
        result.faces(1, i / 3) = indices[i + 2];
        result.faces(2, i / 3) = indices[i + 1];
    }

    return result;
}

template <typename F> IsosurfaceMesh extract_surface(F &func) {
    size_t min_depth = 4;
    occ::timing::StopWatch sw;
    size_t max_depth = func.subdivisions();
    occ::log::debug("minimum subdivisions = {}, maximum subdivisions = {}",
                    min_depth, max_depth);
    auto mc = occ::geometry::mc::LinearHashedMarchingCubes(max_depth);
    mc.min_depth = min_depth;
    occ::log::debug("target separation: {}",
                    (func.side_length() / std::pow(2, max_depth)) *
                        occ::units::BOHR_TO_ANGSTROM);
    occ::log::debug("naive voxel count: {}",
                    std::pow(std::pow(2, max_depth) + 1, 3));
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<uint32_t> faces;
    sw.start();
    mc.extract_with_normals(func, vertices, faces, normals);
    sw.stop();
    double max_calls = std::pow(2, 3 * max_depth);
    occ::log::debug("{} calls ({} % of conventional)", func.num_calls(),
                    (func.num_calls() / max_calls) * 100);
    occ::log::info("Surface extraction took {:.5f} s", sw.read());

    occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                   faces.size() / 3);
    return as_mesh(func, vertices, faces, normals);
}

VertexProperties compute_surface_properties(const Molecule &m1,
                                            const Molecule &m2,
                                            const Eigen::Matrix3Xf &vertices) {
    Eigen::Matrix3Xf inside = m1.positions().cast<float>();
    Eigen::Matrix3Xf outside = m2.positions().cast<float>();
    Eigen::VectorXf vdw_inside = m1.vdw_radii().cast<float>();
    Eigen::VectorXf vdw_outside = m2.vdw_radii().cast<float>();

    VertexProperties properties(vertices.cols());
    int nthreads = occ::parallel::get_num_threads();

    occ::core::KDTree<float> interior_tree(inside.rows(), inside,
                                           occ::core::max_leaf);
    interior_tree.index->buildIndex();

    occ::core::KDTree<float> exterior_tree(outside.rows(), outside,
                                           occ::core::max_leaf);
    exterior_tree.index->buildIndex();

    occ::log::info("Indexes built");
    constexpr size_t num_results = 6;

    auto fill_properties = [&](int thread_id) {
        std::vector<size_t> indices(num_results);
        std::vector<float> dist_sq(num_results);
        std::vector<float> dist_norm(num_results);

        for (int i = 0; i < vertices.cols(); i++) {
            if (i % nthreads != thread_id)
                continue;

            Eigen::Vector3f v = vertices.col(i);
            {

                float dist_inside_norm = std::numeric_limits<float>::max();
                nanoflann::KNNResultSet<float> results(num_results);
                results.init(&indices[0], &dist_sq[0]);
                bool populated = interior_tree.index->findNeighbors(
                    results, v.data(), nanoflann::SearchParams());
                properties.di(i) = std::sqrt(dist_sq[0]);
                properties.di_idx(i) = indices[0];

                size_t inside_idx = 0;
                for (int idx = 0; idx < results.size(); idx++) {

                    float vdw = vdw_inside(indices[idx]);
                    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

                    if (dnorm < dist_inside_norm) {
                        inside_idx = indices[idx];
                        dist_inside_norm = dnorm;
                    }
                }
                properties.di_norm(i) = dist_inside_norm;
                properties.di_norm_idx(i) = inside_idx;
            }

            {
                float dist_outside_norm = std::numeric_limits<float>::max();
                nanoflann::KNNResultSet<float> results(num_results);
                results.init(&indices[0], &dist_sq[0]);
                bool populated = exterior_tree.index->findNeighbors(
                    results, v.data(), nanoflann::SearchParams());
                properties.de(i) = std::sqrt(dist_sq[0]);
                properties.de_idx(i) = indices[0];

                size_t outside_idx = 0;
                for (int idx = 0; idx < results.size(); idx++) {

                    float vdw = vdw_outside(indices[idx]);
                    float dnorm = (std::sqrt(dist_sq[idx]) - vdw) / vdw;

                    if (dnorm < dist_outside_norm) {
                        outside_idx = indices[idx];
                        dist_outside_norm = dnorm;
                    }
                }
                properties.de_norm(i) = dist_outside_norm;
                properties.de_norm_idx(i) = outside_idx;
            }

            properties.dnorm(i) = properties.de_norm(i) + properties.di_norm(i);
        }
    };

    occ::timing::start(occ::timing::category::isosurface_properties);
    occ::parallel::parallel_do(fill_properties);
    occ::timing::stop(occ::timing::category::isosurface_properties);

    return properties;
}

namespace occ::main {

CLI::App *add_isosurface_subcommand(CLI::App &app) {
    CLI::App *iso =
        app.add_subcommand("isosurface", "compute molecular isosurfaces");
    auto config = std::make_shared<IsosurfaceConfig>();

    iso->add_option("geometry", config->geometry_filename,
                    "input geometry file (xyz)")
        ->required();

    iso->add_option("environment", config->environment_filename,
                    "environment geometry file (xyz)");

    iso->add_option("--max-depth", config->max_depth, "Maximum voxel depth");
    iso->add_option("--separation", config->separation,
                    "targt voxel separation");
    iso->add_option("--isovalue", config->isovalue, "target isovalue");
    iso->add_option("--background-density", config->background_density,
                    "add background density to close surface");

    iso->fallthrough();
    iso->callback([config]() { run_isosurface_subcommand(*config); });
    return iso;
}

void run_isosurface_subcommand(IsosurfaceConfig const &config) {
    IsosurfaceMesh mesh;
    VertexProperties properties;

    if (!config.environment_filename.empty()) {
        Molecule m1 = occ::io::molecule_from_xyz_file(config.geometry_filename);
        Molecule m2 =
            occ::io::molecule_from_xyz_file(config.environment_filename);

        occ::log::info("Interior has {} atoms", m1.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m1.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        occ::log::info("Environment has {} atoms", m2.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m2.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = StockholderWeightFunctor(
            m1, m2, config.separation * occ::units::ANGSTROM_TO_BOHR);
        func.set_background_density(config.background_density);
        mesh = extract_surface(func);
        properties = compute_surface_properties(m1, m2, mesh.vertices);
    } else {
        Molecule m = occ::io::molecule_from_xyz_file(config.geometry_filename);
        occ::log::info("Molecule has {} atoms", m.size());

        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}",
                        config.geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = PromoleculeDensityFunctor(
            m, config.separation * occ::units::ANGSTROM_TO_BOHR);
        func.set_isovalue(config.isovalue);
        mesh = extract_surface(func);
    }

    Eigen::Vector3f lower_left = mesh.vertices.rowwise().minCoeff();
    Eigen::Vector3f upper_right = mesh.vertices.rowwise().maxCoeff();
    occ::log::info("Lower corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   lower_left(0), lower_left(1), lower_left(2));
    occ::log::info("Upper corner of mesh: [{:.3f} {:.3f} {:.3f}]",
                   upper_right(0), upper_right(1), upper_right(2));

    occ::log::info("Writing surface to {}", "surface.obj");
    write_obj_file("surface.obj", mesh, properties);
}

} // namespace occ::main
