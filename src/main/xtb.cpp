#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <iostream>
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/io/xyz.h>
#include <occ/xtb/xtb_wrapper.h>

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
    occ::core::Molecule mol = occ::io::molecule_from_xyz_file(input_file);
    occ::xtb::XTBCalculator calc(mol);
    fmt::print("Energy: {}\n", calc.single_point_energy());
    fmt::print("Gradients:\n");
    std::cout << calc.gradients() << '\n';
    return 0;
}
