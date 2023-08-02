#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <chrono>
#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/3rdparty/parallel_hashmap/phmap.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/numpy.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/io/xyz.h>
#include <occ/main/isosurface.h>

namespace fs = std::filesystem;
using occ::IVec;
using occ::Mat3N;
using occ::Vec3;
using occ::core::Element;
using occ::core::Interpolator1D;
using occ::core::Molecule;
using occ::main::PromoleculeDensityFunctor;
using occ::main::StockholderWeightFunctor;

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

template <typename F>
std::pair<Eigen::Matrix3Xf, Eigen::Matrix3Xi>
as_matrices(const F &b, const std::vector<float> &vertices,
            const std::vector<uint32_t> indices) {
    Eigen::Matrix3Xf verts(3, vertices.size() / 3);
    Eigen::Matrix3Xi faces(3, indices.size() / 3);
    float length = b.side_length();
    const auto &origin = b.origin();
    for (size_t i = 0; i < vertices.size(); i += 3) {
        verts(0, i / 3) = vertices[i] * length + origin(0);
        verts(1, i / 3) = vertices[i + 1] * length + origin(1);
        verts(2, i / 3) = vertices[i + 2] * length + origin(2);
    }
    for (size_t i = 0; i < indices.size(); i += 3) {
        faces(0, i / 3) = indices[i];
        faces(1, i / 3) = indices[i + 2];
        faces(2, i / 3) = indices[i + 1];
    }
    verts.array() *= occ::units::BOHR_TO_ANGSTROM;
    return {verts, faces};
}

template <typename F> void extract_surface(F &func) {
    size_t min_depth = 3;
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
    std::vector<uint32_t> faces;
    sw.start();
    mc.extract(func, vertices, faces);
    sw.stop();
    double max_calls = std::pow(2, 3 * max_depth);
    occ::log::debug("{} calls ({} % of conventional)", func.num_calls(),
                    (func.num_calls() / max_calls) * 100);
    occ::log::info("Surface extraction took {:.5f} s", sw.read());

    occ::log::info("Surface has {} vertices, {} faces", vertices.size() / 3,
                   faces.size() / 3);
    auto [v, f] = as_matrices(func, vertices, faces);

    occ::core::numpy::save_npy("verts.npy", v);
    occ::core::numpy::save_npy("faces.npy", f);
    write_ply_file("surface.ply", v, f);
}

int main(int argc, char *argv[]) {
    double minimum_separation = 0.05;
    double maximum_separation = 0.5;

    CLI::App app(
        "occ-hs - A program for Hirshfeld and promolecule surface generation");
    std::string geometry_filename{""}, verbosity{"normal"};
    std::optional<std::string> environment_filename{};
    int threads{1}, max_depth{4};
    double separation{0.2};
    float isovalue{0.02};

    CLI::Option *input_option = app.add_option(
        "geometry_file", geometry_filename, "xyz file of geometry");
    input_option->required();
    app.add_option("environment_file", environment_filename,
                   "xyz file of surroundings for Hirshfeld surface");
    app.add_option("-s,--minimum-separation", minimum_separation,
                   "Minimum separation for surface construction");
    app.add_option("-S,--maximum-separation", maximum_separation,
                   "Maximum separation for surface construction");
    app.add_flag("-t,--threads", threads, "Number of threads");
    app.add_option("--max-depth", max_depth, "Maximum voxel depth");
    app.add_option("--separation", separation, "targt voxel separation");
    app.add_option("--isovalue", isovalue, "targt isovalue");
    // logging verbosity
    app.add_option("-v,--verbosity", verbosity,
                   "logging verbosity {silent,minimal,normal,verbose,debug}");

    CLI11_PARSE(app, argc, argv);

    occ::log::setup_logging(verbosity);

    if (environment_filename) {
        Molecule m1 = occ::io::molecule_from_xyz_file(geometry_filename);
        Molecule m2 = occ::io::molecule_from_xyz_file(*environment_filename);

        occ::log::info("Interior has {} atoms", m1.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}", geometry_filename,
                        "sym", "x", "y", "z");
        for (const auto &atom : m1.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        occ::log::info("Environment has {} atoms", m2.size());
        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}", geometry_filename,
                        "sym", "x", "y", "z");
        for (const auto &atom : m2.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = StockholderWeightFunctor(
            m1, m2, separation * occ::units::ANGSTROM_TO_BOHR);
        extract_surface(func);
    } else {
        Molecule m = occ::io::molecule_from_xyz_file(geometry_filename);
        occ::log::info("Molecule has {} atoms", m.size());

        occ::log::debug("Input geometry {}");
        occ::log::debug("{:3s} {:^10s} {:^10s} {:^10s}", geometry_filename,
                        "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            occ::log::debug("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                            Element(atom.atomic_number).symbol(), atom.x,
                            atom.y, atom.z);
        }

        auto func = PromoleculeDensityFunctor(
            m, separation * occ::units::ANGSTROM_TO_BOHR);
        func.set_isovalue(isovalue);
        extract_surface(func);
    }
    return 0;
}
