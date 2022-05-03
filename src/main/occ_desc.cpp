#include <chrono>
#include <cxxopts.hpp>
#include <filesystem>
#include <occ/core/logger.h>
#include <occ/core/optimize.h>
#include <occ/core/units.h>
#include <occ/io/xyz.h>
#include <occ/slater/slaterbasis.h>

namespace fs = std::filesystem;
using occ::core::Element;
using occ::core::Molecule;

struct SlaterBasis {
    const double buffer = 5.0;
    std::vector<Eigen::Vector3d> coordinates;
    std::vector<occ::slater::Basis> basis;
    const Molecule &molecule;
    float iso = 0.02;
    mutable std::chrono::duration<double> time{0};
    mutable int num_calls{0};

    SlaterBasis(const Molecule &mol) : molecule(mol) {
        auto nums = molecule.atomic_numbers();
        auto pos = molecule.positions();
        pos.array() *= occ::units::ANGSTROM_TO_BOHR;
        auto basis_set = occ::slater::load_slaterbasis("thakkar");
        for (size_t i = 0; i < molecule.size(); i++) {
            int el = nums[i];
            coordinates.push_back(pos.col(i));
            basis.emplace_back(basis_set[Element(el).symbol()]);
        }
    }

    double operator()(double x, double y, double z) const {
        num_calls++;
        auto t1 = std::chrono::high_resolution_clock::now();
        Eigen::Vector3d pos{x, y, z};
        double result = 0.0;
        for (size_t i = 0; i < coordinates.size(); i++) {
            double r = (pos - coordinates[i]).norm();
            result += basis[i].rho(r);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        time += t2 - t1;
        return std::abs(result - iso);
    }

    double dir(double x, double y, double z) const {
        auto lambda = [&](double r) -> double {
            if (r < 0)
                return 1000.0;
            return (*this)(x * r, y * r, z * r);
        };
        std::function<double(double)> func(lambda);
        occ::opt::Brent brent(func);
        brent.set_left(5.0);
        brent.set_right(20.0);
        double xx = brent.xmin();
        double fx = brent.f_xmin();
        fmt::print("Func: {} -> {} ({} calls)\n", xx, fx, brent.num_calls());
        return xx;
    }
};

int main(int argc, char *argv[]) {
    cxxopts::Options options("occ-desc", "Molecular shape descriptors");

    fs::path geometry_filename, environment_filename;
    options.add_options()("i,input", "Input file geometry",
                          cxxopts::value<fs::path>(geometry_filename))(
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

    Molecule m = occ::io::molecule_from_xyz_file(geometry_filename);
    m.translate(-m.center_of_mass());

    fmt::print("Input geometry {}\n{:3s} {:^10s} {:^10s} {:^10s}\n",
               geometry_filename.string(), "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                   Element(atom.atomic_number).symbol(), atom.x, atom.y,
                   atom.z);
    }

    auto basis = SlaterBasis(m);
    fmt::print("Min: {}\n", basis.dir(0.0, 1.0, 0.0));
    fmt::print("Min: {}\n", basis.dir(0.0, 0.0, 1.0));
    fmt::print("Min: {}\n", basis.dir(1.0, 0.0, 0.0));
    return 0;
}
