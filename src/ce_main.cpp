#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>
#include "hf.h"
#include "spinorbital.h"
#include "element.h"
#include "util.h"

using tonto::io::FchkReader;
using tonto::qm::SpinorbitalKind;
using tonto::qm::expectation;
using tonto::chem::Element;
using tonto::util::all_close;

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("ce");
    parser.add_argument("fchk_a").help("Input file fchk A");
    parser.add_argument("fchk_b").help("Input file fchk B");

    parser.add_argument("--model")
            .default_value(std::string("ce-b3lyp"));
    parser.add_argument("-j", "--threads")
            .help("Number of threads")
            .default_value(1)
            .action([](const std::string& value) { return std::stoi(value); });


    tonto::log::set_level(tonto::log::level::debug);
    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }

    using tonto::parallel::nthreads;
    nthreads = parser.get<int>("--threads");
    omp_set_num_threads(nthreads);

    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();

    SpinorbitalKind kind_a = SpinorbitalKind::Restricted;
    SpinorbitalKind kind_b = SpinorbitalKind::Restricted;

    const std::string fchk_filename_a = parser.get<std::string>("fchk_a");
    const std::string fchk_filename_b = parser.get<std::string>("fchk_b");
    FchkReader fchk_a(fchk_filename_a);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_a);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_a, "sym", "x", "y", "z");
    for (const auto &atom : fchk_a.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    fchk_a.basis().print();
    FchkReader fchk_b(fchk_filename_b);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_b);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : fchk_b.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    fchk_b.basis().print();
    tonto::MatRM DA = 0.5 * fchk_a.scf_density_matrix();
    tonto::MatRM DB = 0.5 * fchk_b.scf_density_matrix();
    tonto::log::info("Finished reading SCF density matrices");
    fmt::print("DA:\n{}\nDB:\n{}\n", DA, DB);
    tonto::log::info("Matrices are the same: {}", all_close(DA, DB, 1e-05));
    tonto::hf::HartreeFock hf_a(fchk_a.atoms(), fchk_a.basis_set());
    tonto::hf::HartreeFock hf_b(fchk_b.atoms(), fchk_b.basis_set());

    tonto::MatRM JA, KA, JB, KB, VA, VB, TA, TB, HA, HB;
    VA = hf_a.compute_nuclear_attraction_matrix();
    tonto::log::info("Computed nuclear attraction for B, energy = {}", expectation<SpinorbitalKind::Restricted>(DA, VA));
    TA = hf_a.compute_kinetic_matrix();
    tonto::log::info("Computed kinetic energy for B, energy = {}", expectation<SpinorbitalKind::Restricted>(DA, TA));
    HA = VA + TA;
    tonto::log::info("Computed core hamiltonian for A, energy = {}", expectation<SpinorbitalKind::Restricted>(DA, HA));
    VB = hf_b.compute_nuclear_attraction_matrix();
    tonto::log::info("Computed nuclear attraction for B, energy = {}", expectation<SpinorbitalKind::Restricted>(DB, VB));
    TB = hf_b.compute_kinetic_matrix();
    tonto::log::info("Computed kinetic energy for B, energy = {}", expectation<SpinorbitalKind::Restricted>(DB, TB));
    HB = VB + TB;
    tonto::log::info("Computed core hamiltonian for B, energy = {}", expectation<SpinorbitalKind::Restricted>(DB, HB));
    std::tie(JA, KA) = hf_a.compute_JK(kind_a, DA);
    tonto::log::info("Computed JK matrices for molecule A");
    tonto::log::info("Coulomb: {}, Exchange: {}",
        expectation<SpinorbitalKind::Restricted>(DA, JA),
        expectation<SpinorbitalKind::Restricted>(DA, KA)
    );

    std::tie(JB, KB) = hf_a.compute_JK(kind_b, DB);
    tonto::log::info("Computed JK matrices for molecule B");
    tonto::log::info("Coulomb: {}, Exchange: {}",
        expectation<SpinorbitalKind::Restricted>(DB, JB),
        expectation<SpinorbitalKind::Restricted>(DB, KB)
    );
}
