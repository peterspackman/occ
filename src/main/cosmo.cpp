#include <filesystem>
#include <tonto/core/logger.h>
#include <tonto/core/molecule.h>
#include <tonto/core/timings.h>
#include <tonto/geometry/linear_hashed_marching_cubes.h>
#include <tonto/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/os.h>
#include <tonto/slater/thakkar.h>

namespace fs = std::filesystem;
using tonto::chem::Molecule;
using tonto::chem::Element;
using tonto::geometry::mc::LinearHashedMarchingCubes;
using tonto::geometry::mc::MarchingCubes;
using tonto::timing::StopWatch;

struct InputConfiguration {
    fs::path geometry_filename;
};

struct SlaterBasis {
    const double buffer = 3.8;
    std::vector<Eigen::Vector3d> coordinates;
    std::vector<tonto::slater::Basis> basis;
    const Molecule& molecule;
    Eigen::Vector3d origin{0, 0, 0};
    Eigen::Vector3d lengths{0, 0, 0};
    float iso = 0.002;
    mutable int num_calls{0};

    SlaterBasis(const Molecule &mol) : molecule(mol)
    {
        auto nums = molecule.atomic_numbers();
        auto pos = molecule.positions();
        Eigen::Vector3d minp = pos.rowwise().minCoeff();
        Eigen::Vector3d maxp = pos.rowwise().maxCoeff();
        minp.array() -= buffer;
        maxp.array() += buffer;
        for(size_t i = 0; i < molecule.size(); i++)
        {
            int el = nums[i];
            coordinates.push_back(pos.col(i));
            basis.emplace_back(tonto::thakkar::basis_for_element(el));
        }
        origin = minp;
        lengths = maxp - minp;
        fmt::print("Bottom left\n{}\nlengths\n{}\n", origin, lengths);
    }

    float operator()(float x, float y, float z) const
    {
        num_calls++;
        float result = 0.0;
        Eigen::Vector3d pos{
            static_cast<double>(x * lengths(0) + origin(0)),
            static_cast<double>(y * lengths(1) + origin(1)),
            static_cast<double>(z * lengths(2) + origin(2))
        };
        for(size_t i = 0; i < coordinates.size(); i++)
        {
            double r = (pos - coordinates[i]).norm();
            result += static_cast<float>(basis[i].rho(r));
        }
        return iso - result;
    }
};

struct SASA {
    Eigen::Matrix3Xf coordinates;
    Eigen::VectorXf radii;
    const Molecule& molecule;
    Eigen::Vector3f origin{0, 0, 0};
    Eigen::Vector3f lengths{0, 0, 0};
    float probe_radius = 1.8;
    mutable int num_calls{0};

    SASA(const Molecule &mol) : molecule(mol)
    {
        radii = molecule.vdw_radii().cast<float>();
        coordinates = molecule.positions().cast<float>();
        float buffer = probe_radius + radii.maxCoeff() + 0.1;
        Eigen::Vector3f minp = coordinates.rowwise().minCoeff();
        Eigen::Vector3f maxp = coordinates.rowwise().maxCoeff();
        minp.array() -= buffer;
        maxp.array() += buffer;
        origin = minp;
        lengths = maxp - minp;
        fmt::print("Buffer:{}\nBottom left\n{}\nlengths\n{}\n", buffer,origin, lengths);
    }

    float operator()(float x, float y, float z) const
    {
        num_calls++;
        float result = 10000.0;
        Eigen::Vector3f pos{
            x * lengths(0) + origin(0),
            y * lengths(1) + origin(1),
            z * lengths(2) + origin(2)
        };
        for(size_t i = 0; i < coordinates.size(); i++)
        {
            float r = (pos - coordinates.col(i)).norm();
            result = std::min(result, r - radii(i) - probe_radius);
        }
        return result;
    }

};

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("tonto");
    parser.add_argument("input").help("Input file geometry");
    tonto::log::set_level(tonto::log::level::debug);
    spdlog::set_level(spdlog::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }

    InputConfiguration config;
    config.geometry_filename = parser.get<std::string>("input");

    Molecule m = tonto::chem::read_xyz_file(config.geometry_filename);

    fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n", config.geometry_filename, "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    
    StopWatch<1> sw;
    SlaterBasis b(m);
    float resolution = 0.2;
    size_t divisions = std::ceil(b.lengths.maxCoeff() / resolution);
    fmt::print("Divisions: {}\n", divisions);
    MarchingCubes mc(divisions);

    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    sw.start(0);
    mc.extract(b, vertices, indices);
    sw.stop(0);
    fmt::print("{} vertices, {} faces in {}\n", vertices.size() / 3, indices.size() / 3, sw.read(0));
    auto verts = fmt::output_file("verts.txt");
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts.print("{:20.12f} {:20.12f} {:20.12f}\n",
                    vertices[i] * b.lengths(0) + b.origin(0),
                    vertices[i + 1] * b.lengths(1) + b.origin(1),
                    vertices[i + 2] * b.lengths(2) + b.origin(2));
    }
    auto faces = fmt::output_file("faces.txt");
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces.print("{:12d} {:12d} {:12d}\n", indices[i], indices[i + 2], indices[i + 1]);
    }
    fmt::print("{} function calls\n", b.num_calls);


    return 0;
}
