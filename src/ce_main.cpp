#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>
#include "hf.h"
#include "spinorbital.h"
#include "element.h"
#include "util.h"
#include "gto.h"
#include "pairinteraction.h"

using tonto::io::FchkReader;
using tonto::qm::SpinorbitalKind;
using tonto::qm::expectation;
using tonto::chem::Element;
using tonto::util::all_close;
using tonto::util::join;
using tonto::gto::shell_component_labels;
using tonto::MatRM;
using tonto::Vec;
using tonto::interaction::merge_molecular_orbitals;
using tonto::interaction::merge_basis_sets;
using tonto::interaction::merge_atoms;
using tonto::qm::BasisSet;

struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
};

struct Wavefunction {
    BasisSet basis;
    int num_occ;
    int num_electrons;
    std::vector<libint2::Atom> atoms;
    MatRM C, C_occ, D, T, V, H, J, K;
    Vec mo_energies;
    Energy energy;
};

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

    FchkReader fchk_b(fchk_filename_b);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_b);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : fchk_b.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    Wavefunction A, B, AB;
    A.num_occ = fchk_a.num_alpha();
    B.num_occ = fchk_b.num_alpha();

    A.basis = fchk_a.basis_set();
    B.basis = fchk_b.basis_set();
    A.atoms = fchk_a.atoms();
    B.atoms = fchk_b.atoms();

    A.C = fchk_a.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(A.basis, A.C);
    A.C_occ = A.C.leftCols(A.num_occ);
    A.D = A.C_occ * A.C_occ.transpose();

    B.C = fchk_b.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(B.basis, B.C);
    B.C_occ = B.C.leftCols(B.num_occ);
    B.D = B.C_occ * B.C_occ.transpose();

    tonto::log::info("Finished reading SCF density matrices");
    tonto::log::info("Matrices are the same: {}", all_close(A.D, B.D, 1e-05));
    tonto::hf::HartreeFock hf_a(A.atoms, A.basis);
    tonto::hf::HartreeFock hf_b(B.atoms, B.basis);

    A.V = hf_a.compute_nuclear_attraction_matrix();
    A.energy.nuclear_attraction = expectation<SpinorbitalKind::Restricted>(A.D, A.V);

    tonto::log::info("Computed nuclear attraction for A, energy = {}", A.energy.nuclear_attraction);
    A.T = hf_a.compute_kinetic_matrix();
    A.energy.kinetic = expectation<SpinorbitalKind::Restricted>(A.D, A.T);
    tonto::log::info("Computed kinetic energy for A, energy = {}", A.energy.kinetic);
    A.H = A.V + A.T;
    A.energy.core = expectation<SpinorbitalKind::Restricted>(A.D, A.H);
    tonto::log::info("Computed core hamiltonian for A, energy = {}", A.energy.core);

    B.V = hf_b.compute_nuclear_attraction_matrix();
    B.energy.nuclear_attraction = expectation<SpinorbitalKind::Restricted>(B.D, B.V);
    tonto::log::info("Computed nuclear attraction for B, energy = {}", B.energy.nuclear_attraction);
    B.T = hf_b.compute_kinetic_matrix();
    B.energy.kinetic = expectation<SpinorbitalKind::Restricted>(B.D, B.T);
    tonto::log::info("Computed kinetic energy for B, energy = {}", B.energy.kinetic);
    B.H = B.V + B.T;
    B.energy.core = expectation<SpinorbitalKind::Restricted>(B.D, B.H);
    tonto::log::info("Computed core hamiltonian for B, energy = {}", B.energy.core);

    std::tie(A.J, A.K) = hf_a.compute_JK(kind_a, A.D);
    tonto::log::info("Computed JK matrices for molecule A");
    A.energy.coulomb = expectation<SpinorbitalKind::Restricted>(A.D, A.J);
    A.energy.exchange = expectation<SpinorbitalKind::Restricted>(A.D, A.K);
    tonto::log::info("Coulomb: {}, Exchange: {}", A.energy.coulomb, A.energy.exchange);

    std::tie(B.J, B.K) = hf_b.compute_JK(kind_b, B.D);
    tonto::log::info("Computed JK matrices for molecule B");
    B.energy.coulomb = expectation<SpinorbitalKind::Restricted>(B.D, B.J);
    B.energy.exchange = expectation<SpinorbitalKind::Restricted>(B.D, B.K);
    tonto::log::info("Coulomb: {}, Exchange: {}", B.energy.coulomb, B.energy.exchange);

    A.mo_energies = fchk_a.alpha_mo_energies();
    B.mo_energies = fchk_b.alpha_mo_energies();
    std::tie(AB.C, AB.mo_energies) = merge_molecular_orbitals(A.C, B.C, A.mo_energies, B.mo_energies);
    fmt::print("MO A: {}\n{}\n", A.mo_energies.transpose(), A.C);
    fmt::print("MO B: {}\n{}\n", B.mo_energies.transpose(), B.C);
    fmt::print("MO AB: {}\n{}\n", AB.mo_energies.transpose(), AB.C);
    AB.atoms = merge_atoms(A.atoms, B.atoms);

    fmt::print("Merged geometry\n{:3s} {:^10s} {:^10s} {:^10s}\n", "sym", "x", "y", "z");
    for (const auto &atom : AB.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    AB.basis = merge_basis_sets(A.basis, B.basis);
    auto hf_AB = tonto::hf::HartreeFock(AB.atoms, AB.basis);
    MatRM S_AB = hf_AB.compute_overlap_matrix();
    fmt::print("S\n{}\n", S_AB);
    MatRM X_AB, X_inv_AB;
    double X_AB_condition_number;
    double S_condition_number_threshold = 1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X_AB, X_inv_AB, X_AB_condition_number) = tonto::conditioning_orthogonalizer(S_AB, S_condition_number_threshold);
    fmt::print("Condition number: {}\n", X_AB_condition_number);

    fmt::print("X\n{}\n", X_AB);
    fmt::print("X_inv\n{}\n", X_inv_AB);
    AB.C =  AB.C * X_inv_AB;
    AB.num_occ = A.num_occ + B.num_occ;
    AB.C_occ = AB.C.leftCols(AB.num_occ);
    AB.D= AB.C_occ * AB.C_occ.transpose();
    fmt::print("DAB\n{}\n", AB.D);
    auto kind_ab = kind_a;
    std::tie(AB.J, AB.K) = hf_AB.compute_JK(kind_ab, AB.D);
    AB.energy.coulomb = expectation<SpinorbitalKind::Restricted>(AB.D, AB.J);
    AB.energy.exchange = expectation<SpinorbitalKind::Restricted>(AB.D, AB.K);
    tonto::log::info("Coulomb: {}, Exchange: {}", AB.energy.coulomb, AB.energy.exchange);

    AB.V = hf_AB.compute_nuclear_attraction_matrix();
    AB.energy.nuclear_attraction = expectation<SpinorbitalKind::Restricted>(AB.D, AB.V);
    tonto::log::info("Computed nuclear attraction for AB, energy = {}", AB.energy.nuclear_attraction);
    AB.T = hf_AB.compute_kinetic_matrix();
    AB.energy.kinetic = expectation<SpinorbitalKind::Restricted>(AB.D, AB.T);
    tonto::log::info("Computed kinetic energy for AB, energy = {}", AB.energy.kinetic);
    AB.H = AB.V + AB.T;
    AB.energy.core = expectation<SpinorbitalKind::Restricted>(AB.D, AB.H);
    tonto::log::info("Computed core hamiltonian for B, energy = {}", AB.energy.core);
    fmt::print("Differences:\nEcoul = {}\nEexch = {}\nEnuc  = {}\nEkin  = {}\nEcore = {}\n",
               AB.energy.coulomb - A.energy.coulomb - B.energy.coulomb,
               AB.energy.exchange - A.energy.exchange - B.energy.exchange,
               AB.energy.nuclear_attraction - A.energy.nuclear_attraction - B.energy.nuclear_attraction,
               AB.energy.kinetic - A.energy.kinetic - B.energy.kinetic,
               AB.energy.core - A.energy.core - B.energy.core);

}
