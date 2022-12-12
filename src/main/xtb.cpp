#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <LBFGS.h>
#include <filesystem>
#include <iostream>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/io/cifparser.h>
#include <occ/io/xyz.h>
#include <occ/xtb/xtb_wrapper.h>

namespace fs = std::filesystem;
using LBFGSpp::LBFGSParam;
using LBFGSpp::LBFGSSolver;

inline Eigen::Vector<double, 6> to_voigt(const occ::Mat3 &mat) {
    Eigen::Vector<double, 6> v;
    v(0) = mat(0, 0);
    v(1) = mat(1, 1);
    v(2) = mat(2, 2);

    v(3) = mat(1, 2);

    v(4) = mat(0, 2);
    v(5) = mat(0, 1);
    return v;
}

inline occ::Mat3 from_voigt(const Eigen::Vector<double, 6> &v) {
    occ::Mat3 result;
    result(0, 0) = v(0);
    result(1, 1) = v(1);
    result(2, 2) = v(2);

    result(1, 2) = v(3);
    result(2, 1) = v(3);

    result(0, 2) = v(4);
    result(2, 0) = v(4);

    result(0, 1) = v(5);
    result(1, 0) = v(5);
    return result;
}

int main(int argc, char *argv[]) {

    occ::timing::start(occ::timing::category::global);
    occ::timing::start(occ::timing::category::io);
    CLI::App app("occ - A program for quantum chemistry");
    std::string input_file{""}, method{"gfn2-xtb"}, verbosity{"warn"};
    int threads{1}, charge{0}, multiplicity{1};

    CLI::Option *input_option =
        app.add_option("input", input_file, "input file");
    input_option->required();
    app.add_option("-t,--threads", threads, "number of threads");
    app.add_option("-c,--charge", charge, "system net charge");
    app.add_option("--multiplicity", multiplicity, "system multiplicity");
    app.add_option("-v,--verbosity", verbosity,
                   "logging verbosity {silent,minimal,normal,verbose,debug}");

    CLI11_PARSE(app, argc, argv);
    occ::log::setup_logging(verbosity);
    occ::timing::stop(occ::timing::category::io);

    occ::parallel::set_num_threads(std::max(1, threads));

    fmt::print("TBLITE VERSION: {}\n", occ::xtb::tblite_version());
    occ::xtb::XTBCalculator calc = [&]() {
        auto path = fs::path(input_file);
        std::string ext = path.extension().string();
        if (ext == ".xyz") {
            occ::core::Molecule mol =
                occ::io::molecule_from_xyz_file(input_file);
            return occ::xtb::XTBCalculator(mol);
        } else {
            occ::io::CifParser parser;
            occ::crystal::Crystal crystal =
                parser.parse_crystal(input_file).value();
            return occ::xtb::XTBCalculator(crystal);
        }
    }();

    int natom = calc.num_atoms();

    auto func = [&](const occ::Vec &posvec, occ::Vec &gradvec) {
        occ::Mat3N new_positions = posvec.topRows(3 * natom).reshaped(3, natom);
        occ::Mat3 new_vectors = from_voigt(posvec.bottomRows(6));
        fmt::print("New vectors:\n{}\n", new_vectors);
        calc.update_structure(new_positions, new_vectors);
        double e = calc.single_point_energy();
        gradvec.topRows(3 * natom) = calc.gradients().reshaped();
        gradvec.bottomRows(6) =
            -to_voigt(calc.virial()) / calc.lattice_vectors().determinant();
        fmt::print("Cell grad:\n{}\n", gradvec.bottomRows(6).transpose());
        return e;
    };

    LBFGSParam<double> param;
    param.epsilon = 1e-3;
    param.max_iterations = 200;
    param.max_linesearch = 100;
    LBFGSSolver<double> solver(param);

    occ::Vec initial(3 * (natom + 2));
    initial.topRows(3 * natom) = calc.positions().reshaped();
    initial.bottomRows(6) = to_voigt(calc.lattice_vectors());
    fmt::print("posvec:\n{}\n", initial);

    double fx;
    int niter = solver.minimize(func, initial, fx);

    fmt::print("Final energy after {} iterations: {}\n", niter,
               calc.single_point_energy());
    fmt::print("Final gradient norm: {}\n", calc.gradients().norm());

    occ::crystal::Crystal result = calc.to_crystal();
    fmt::print("Final lattice parameters:\n");
    fmt::print("a: {}\n", result.unit_cell().a());
    fmt::print("b: {}\n", result.unit_cell().b());
    fmt::print("c: {}\n", result.unit_cell().c());
    fmt::print("alpha: {}\n", occ::units::degrees(result.unit_cell().alpha()));
    fmt::print("beta: {}\n", occ::units::degrees(result.unit_cell().beta()));
    fmt::print("gamma: {}\n", occ::units::degrees(result.unit_cell().gamma()));

    fmt::print("Numbers\n{}\n", result.asymmetric_unit().atomic_numbers);
    fmt::print("Positions\n{}\n",
               result.asymmetric_unit().positions.transpose());

    return 0;
}
