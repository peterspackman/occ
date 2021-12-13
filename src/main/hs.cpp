#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/ostream.h>
#include <occ/io/xyz.h>
#include <occ/3rdparty/robin_hood.h>
#include <occ/core/eigenp.h>
#include <occ/core/interpolator.h>
#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/geometry/linear_hashed_marching_cubes.h>
#include <occ/geometry/marching_cubes.h>
#include <occ/slater/slaterbasis.h>

namespace fs = std::filesystem;
using occ::core::Element;
using occ::core::Molecule;
using occ::core::Interpolator1D;

struct HirshfeldBasis {
    std::vector<Eigen::Vector3d> coordinates;
    double buffer = 6.0;
    const Molecule &interior, exterior;
    Eigen::Vector3d origin{0, 0, 0};
    size_t num_interior{0};
    double length{0.0};
    float iso = 0.5;
    float fac = 0.5;
    mutable int num_calls{0};
    std::vector<int> elements;
    robin_hood::unordered_flat_map<
        int, Interpolator1D<double, occ::core::DomainMapping::Log>>
        interpolators;

    HirshfeldBasis(const Molecule &in, Molecule &ext)
        : interior(in), exterior(ext) {
        auto nums_in = in.atomic_numbers();
        auto pos_in = in.positions();
        auto nums_ext = ext.atomic_numbers();
        auto pos_ext = ext.positions();
        pos_in.array() *= occ::units::ANGSTROM_TO_BOHR;
        pos_ext.array() *= occ::units::ANGSTROM_TO_BOHR;
        Eigen::Vector3d minp_in = pos_in.rowwise().minCoeff();
        Eigen::Vector3d maxp_in = pos_in.rowwise().maxCoeff();
        Eigen::Vector3d minp_ext = pos_ext.rowwise().minCoeff();
        Eigen::Vector3d maxp_ext = pos_ext.rowwise().maxCoeff();

        num_interior = in.size();
        auto basis = occ::slater::load_slaterbasis("thakkar");
        for (size_t i = 0; i < in.size(); i++) {
            int el = nums_in[i];
            coordinates.push_back(pos_in.col(i));
            elements.push_back(el);
            auto search = interpolators.find(el);
            if (search == interpolators.end()) {
                auto b = basis[Element(el).symbol()];
                auto func = [&b](double x) { return b.rho(x); };
                interpolators[el] =
                    Interpolator1D<double, occ::core::DomainMapping::Log>(
                        func, 0.1, 20.0, 4096);
            }
        }

        for (size_t i = 0; i < ext.size(); i++) {
            int el = nums_ext[i];
            elements.push_back(el);
            coordinates.push_back(pos_ext.col(i));
            auto search = interpolators.find(el);
            if (search == interpolators.end()) {
                auto b = basis[Element(el).symbol()];
                auto func = [&b](double x) { return b.rho(x); };
                interpolators[el] =
                    Interpolator1D<double, occ::core::DomainMapping::Log>(
                        func, 0.1, 20.0, 4096);
            }
        }
        origin = minp_in;
        origin.array() -= buffer;
        length = (maxp_in - origin).maxCoeff() + buffer;
        fac = 1.0 / length;
        fmt::print("Bottom left\n{}\nlength\n{}\n", origin, length);
    }

    float operator()(float x, float y, float z) const {
        num_calls++;
        float tot_i{0.0}, tot_e{0.0};
        Eigen::Vector3d pos{static_cast<double>(x * length + origin(0)),
                            static_cast<double>(y * length + origin(1)),
                            static_cast<double>(z * length + origin(2))};
        for (size_t i = 0; i < coordinates.size(); i++) {
            int el = elements[i];
            double r = (pos - coordinates[i]).norm();
            auto interp = interpolators.at(el);
            double rho = static_cast<float>(interp(r));
            if (i >= num_interior)
                tot_e += rho;
            else
                tot_i += rho;
        }
        double v = iso - tot_i / (tot_i + tot_e);
        return fac * v;
    }
};

struct SlaterBasis {
    const double buffer = 5.0;
    std::vector<Eigen::Vector3d> coordinates;
    std::vector<occ::slater::Basis> basis;
    const Molecule &molecule;
    Eigen::Vector3d origin{0, 0, 0};
    double length{0.0};
    float iso = 0.02;
    float fac = 0.5;
    mutable int num_calls{0};

    SlaterBasis(const Molecule &mol) : molecule(mol) {
        auto nums = molecule.atomic_numbers();
        auto pos = molecule.positions();
        pos.array() *= occ::units::ANGSTROM_TO_BOHR;
        Eigen::Vector3d minp = pos.rowwise().minCoeff();
        Eigen::Vector3d maxp = pos.rowwise().maxCoeff();
        minp.array() -= buffer;
        maxp.array() += buffer;
        auto basis_set = occ::slater::load_slaterbasis("thakkar");
        for (size_t i = 0; i < molecule.size(); i++) {
            int el = nums[i];
            coordinates.push_back(pos.col(i));
            basis.emplace_back(basis_set[Element(el).symbol()]);
        }
        origin = minp;
        length = (maxp - minp).maxCoeff();
        fac = 0.25 / length;
        fmt::print("Bottom left\n{}\nlength\n{}\n", origin, length);
    }

    float operator()(float x, float y, float z) const {
        num_calls++;
        float result = 0.0;
        Eigen::Vector3d pos{static_cast<double>(x * length + origin(0)),
                            static_cast<double>(y * length + origin(1)),
                            static_cast<double>(z * length + origin(2))};
        for (size_t i = 0; i < coordinates.size(); i++) {
            double r = (pos - coordinates[i]).norm();
            result += static_cast<float>(basis[i].rho(r));
        }
        return fac * (iso - result);
    }
};

std::pair<Eigen::Matrix3Xf, Eigen::Matrix3Xi>
as_matrices(const SlaterBasis &b, const std::vector<float> &vertices,
            const std::vector<uint32_t> indices) {
    Eigen::Matrix3Xf verts(3, vertices.size() / 3);
    Eigen::Matrix3Xi faces(3, indices.size() / 3);
    for (size_t i = 0; i < vertices.size(); i += 3) {
        verts(0, i / 3) = vertices[i] * b.length + b.origin(0);
        verts(1, i / 3) = vertices[i + 1] * b.length + b.origin(1);
        verts(2, i / 3) = vertices[i + 2] * b.length + b.origin(2);
    }
    for (size_t i = 0; i < indices.size(); i += 3) {
        faces(0, i / 3) = indices[i];
        faces(1, i / 3) = indices[i + 2];
        faces(2, i / 3) = indices[i + 1];
    }
    return {verts, faces};
}

std::pair<Eigen::Matrix3Xf, Eigen::Matrix3Xi>
as_matrices(const HirshfeldBasis &b, const std::vector<float> &vertices,
            const std::vector<uint32_t> indices) {
    Eigen::Matrix3Xf verts(3, vertices.size() / 3);
    Eigen::Matrix3Xi faces(3, indices.size() / 3);
    for (size_t i = 0; i < vertices.size(); i += 3) {
        verts(0, i / 3) = vertices[i] * b.length + b.origin(0);
        verts(1, i / 3) = vertices[i + 1] * b.length + b.origin(1);
        verts(2, i / 3) = vertices[i + 2] * b.length + b.origin(2);
    }
    for (size_t i = 0; i < indices.size(); i += 3) {
        faces(0, i / 3) = indices[i];
        faces(1, i / 3) = indices[i + 2];
        faces(2, i / 3) = indices[i + 1];
    }
    return {verts, faces};
}

int main(int argc, char *argv[]) {
    cxxopts::Options options("occ-hs", "Surface generation program");
    double minimum_separation = 0.05;
    double maximum_separation = 0.5;

    fs::path geometry_filename, environment_filename;
    options.add_options()("i,input", "Input file geometry",
                          cxxopts::value<fs::path>(geometry_filename))(
        "s,minimum-separation", "Minimum separation",
        cxxopts::value<double>(minimum_separation))(
        "S,maximum-separation", "Maximum separation",
        cxxopts::value<double>(maximum_separation))(
        "e,environment", "Environment geometry for HS",
        cxxopts::value<fs::path>(environment_filename))(
        "t,threads", "Number of threads",
        cxxopts::value<int>()->default_value("1"));

    options.parse_positional({"input", "environment"});
    occ::log::set_level(occ::log::level::debug);
    spdlog::set_level(spdlog::level::debug);

    try {
        auto result = options.parse(argc, argv);
    } catch (const std::runtime_error &err) {
        occ::log::error("error when parsing command line arguments: {}",
                        err.what());
        fmt::print("{}\n", options.help());
        exit(1);
    }

    if (!environment_filename.empty()) {
        Molecule m1 = occ::io::molecule_from_xyz_file(geometry_filename);
        Molecule m2 = occ::io::molecule_from_xyz_file(environment_filename);

        fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n",
                   geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m1.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                       Element(atom.atomic_number).symbol(), atom.x, atom.y,
                       atom.z);
        }

        fmt::print("Environment geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n",
                   geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m2.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                       Element(atom.atomic_number).symbol(), atom.x, atom.y,
                       atom.z);
        }

        auto basis = HirshfeldBasis(m1, m2);
        size_t max_depth = 7;
        size_t min_depth = 3;
        fmt::print("Min depth = {}, max depth = {}\n", min_depth, max_depth);
        auto mc = occ::geometry::mc::LinearHashedMarchingCubes(max_depth);
        mc.min_depth = min_depth;
        std::vector<float> vertices;
        std::vector<uint32_t> faces;
        mc.extract(basis, vertices, faces);
        double max_calls = std::pow(2, 3 * max_depth);
        fmt::print("{} calls ({} % of conventional)\n", basis.num_calls,
                   (basis.num_calls / max_calls) * 100);

        auto [v, f] = as_matrices(basis, vertices, faces);

        enpy::save_npy("verts.npy", v);
        enpy::save_npy("faces.npy", f);

    } else {
        Molecule m = occ::io::molecule_from_xyz_file(geometry_filename);

        fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n",
                   geometry_filename, "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                       Element(atom.atomic_number).symbol(), atom.x, atom.y,
                       atom.z);
        }

        auto basis = SlaterBasis(m);
        size_t max_depth = 7;
        size_t min_depth = 1;
        fmt::print("Min depth = {}, max depth = {}\n", min_depth, max_depth);
        auto mc = occ::geometry::mc::LinearHashedMarchingCubes(max_depth);
        mc.min_depth = min_depth;
        std::vector<float> vertices;
        std::vector<uint32_t> faces;
        mc.extract(basis, vertices, faces);
        double max_calls = std::pow(2, 3 * max_depth);
        fmt::print("{} calls ({} % of conventional)\n", basis.num_calls,
                   (basis.num_calls / max_calls) * 100);

        auto [v, f] = as_matrices(basis, vertices, faces);

        enpy::save_npy("verts.npy", v);
        enpy::save_npy("faces.npy", f);
    }
    return 0;
}
