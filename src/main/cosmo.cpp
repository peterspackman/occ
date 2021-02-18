#include <filesystem>
#include <tonto/core/logger.h>
#include <tonto/core/molecule.h>
#include <tonto/core/timings.h>
#include <tonto/core/eem.h>
#include <tonto/geometry/linear_hashed_marching_cubes.h>
#include <tonto/3rdparty/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fmt/os.h>
#include <tonto/slater/thakkar.h>
#include <tonto/solvent/cosmo.h>
namespace fs = std::filesystem;
using tonto::chem::Molecule;
using tonto::chem::Element;
using tonto::geometry::mc::LinearHashedMarchingCubes;
using tonto::geometry::mc::MarchingCubes;
using tonto::timing::StopWatch;
using tonto::solvent::COSMO;

struct InputConfiguration {
    fs::path geometry_filename;
};

struct SlaterBasis {
    const double buffer = 4.8;
    std::vector<Eigen::Vector3d> coordinates;
    std::vector<tonto::slater::Basis> basis;
    const Molecule& molecule;
    Eigen::Vector3d origin{0, 0, 0};
    Eigen::Vector3d lengths{0, 0, 0};
    float iso = 0.0002;
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

std::pair<Eigen::Matrix3Xf, Eigen::Matrix3Xi> as_matrices(const SlaterBasis& b, const std::vector<float>& vertices, const std::vector<uint32_t> indices)
{
    Eigen::Matrix3Xf verts(3, vertices.size() / 3); 
    Eigen::Matrix3Xi faces(3, indices.size() / 3);
    for(size_t i = 0; i < vertices.size(); i += 3)
    {
        verts(0, i/3) = vertices[i] * b.lengths(0) + b.origin(0);
        verts(1, i/3) = vertices[i + 1] * b.lengths(1) + b.origin(1);
        verts(2, i/3) = vertices[i + 2] * b.lengths(2) + b.origin(2);
    }
    for(size_t i = 0; i < indices.size(); i += 3)
    {
        faces(0, i/3) = indices[i];
        faces(1, i/3) = indices[i + 2];
        faces(2, i/3) = indices[i + 1];
    }
    return {verts, faces};
}

Eigen::VectorXf calculate_areas(const Eigen::Matrix3Xf &verts, const Eigen::Matrix3Xi &faces)
{
    Eigen::VectorXf face_areas(faces.cols());
    for(size_t i = 0; i < faces.cols(); i++)
    {
        const auto a = verts.col(faces(0, i));
        const auto b = verts.col(faces(1, i));
        const auto c = verts.col(faces(2, i));
        face_areas(i) = ((a - c).cross(b - c)).norm() * 0.5;
    }

    fmt::print("Surface area (faces): {}\n", face_areas.sum());

    Eigen::VectorXf vertex_areas = Eigen::VectorXf::Zero(verts.cols());
    Eigen::VectorXf vertex_counts = Eigen::VectorXf::Zero(verts.cols());

    for(size_t i = 0; i < faces.cols(); i++)
    {
        vertex_areas(faces(0, i)) += face_areas(i);
        vertex_areas(faces(1, i)) += face_areas(i);
        vertex_areas(faces(2, i)) += face_areas(i);
    }

    vertex_areas.array() /= 3;
    return vertex_areas;
}

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
    float resolution = 0.7;
    size_t divisions = std::ceil(b.lengths.maxCoeff() / resolution);
    fmt::print("Divisions: {}\n", divisions);
    MarchingCubes mc(divisions);

    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    sw.start(0);
    mc.extract(b, vertices, indices);
    sw.stop(0);

    Eigen::Matrix3Xf verts;
    Eigen::Matrix3Xi faces;
    std::tie(verts, faces) = as_matrices(b, vertices, indices);

    fmt::print("{} vertices, {} faces in {}\n", verts.cols(), faces.cols(), sw.read(0));
    fmt::print("{} function calls\n", b.num_calls);
    auto areas = calculate_areas(verts, faces);
    tonto::Vec charges = tonto::Vec::Zero(areas.rows());
    tonto::Vec areasd = areas.cast<double>();
    tonto::Mat3N points = verts.cast<double>();
    charges.array() = -0.005;

    COSMO cosmo(78.40);
    auto result = cosmo(points, areasd, charges);
    auto vout = fmt::output_file("verts.txt");
    vout.print("{}", verts.transpose());
    auto fout = fmt::output_file("faces.txt");
    fout.print("{}", faces.transpose());
    fmt::print("Surface area: {}\n", areas.sum());

    tonto::Mat3N pos = m.positions();
    tonto::IVec nums = m.atomic_numbers();
    tonto::Vec partial_charges = tonto::core::charges::eem_partial_charges(nums, pos);
    fmt::print("Charges:\n{}\n", partial_charges);
    return 0;
}
