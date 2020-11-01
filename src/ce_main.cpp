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

template<SpinorbitalKind kind>
void compute_energies(Wavefunction& wfn, tonto::hf::HartreeFock& hf)
{
    double exchange_factor = - 0.5;
    if constexpr(kind != SpinorbitalKind::Restricted) exchange_factor = - 1.0;
    wfn.V = hf.compute_nuclear_attraction_matrix();
    wfn.energy.nuclear_attraction = expectation<kind>(wfn.D, wfn.V);
    wfn.T = hf.compute_kinetic_matrix();
    wfn.energy.kinetic = expectation<kind>(wfn.D, wfn.T);
    wfn.H = wfn.V + wfn.T;
    wfn.energy.core = expectation<kind>(wfn.D, wfn.H);
    std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.D);
    wfn.energy.coulomb = expectation<kind>(wfn.D, wfn.J);
    wfn.energy.exchange = exchange_factor * expectation<kind>(wfn.D, wfn.K);
    wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
}


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

    const SpinorbitalKind kind_a = SpinorbitalKind::Restricted;
    const SpinorbitalKind kind_b = SpinorbitalKind::Restricted;

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

    Wavefunction A, B, ABn, ABo;
    A.num_occ = fchk_a.num_alpha();
    B.num_occ = fchk_b.num_alpha();

    A.basis = fchk_a.basis_set();
    B.basis = fchk_b.basis_set();
    A.atoms = fchk_a.atoms();
    B.atoms = fchk_b.atoms();

    double exchange_factor = 0.5;

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

    A.mo_energies = fchk_a.alpha_mo_energies();
    B.mo_energies = fchk_b.alpha_mo_energies();

    compute_energies<kind_a>(A, hf_a);
    compute_energies<kind_b>(B, hf_b);

    std::tie(ABn.C, ABn.mo_energies) = merge_molecular_orbitals(A.C, B.C, A.mo_energies, B.mo_energies);
    fmt::print("MO A: {}\n{}\n", A.mo_energies.transpose(), A.C);
    fmt::print("MO B: {}\n{}\n", B.mo_energies.transpose(), B.C);
    fmt::print("MO ABn: {}\n{}\n", ABn.mo_energies.transpose(), ABn.C);
    ABn.atoms = merge_atoms(A.atoms, B.atoms);
    ABo.atoms = ABn.atoms;

    fmt::print("Merged geometry\n{:3s} {:^10s} {:^10s} {:^10s}\n", "sym", "x", "y", "z");
    for (const auto &atom : ABn.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    ABn.basis = merge_basis_sets(A.basis, B.basis);
    auto hf_AB = tonto::hf::HartreeFock(ABn.atoms, ABn.basis);


    MatRM S_AB = hf_AB.compute_overlap_matrix();
    fmt::print("S\n{}\n", S_AB);
    MatRM X_AB, X_inv_AB;
    double X_AB_condition_number;
    double S_condition_number_threshold = 1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X_AB, X_inv_AB, X_AB_condition_number) = tonto::conditioning_orthogonalizer(S_AB, S_condition_number_threshold);
    fmt::print("Condition number: {}\n", X_AB_condition_number);

    fmt::print("X\n{}\n", X_AB);
    fmt::print("X_inv\n{}\n", X_inv_AB);
    ABo.C =  ABn.C * X_inv_AB;
    ABn.num_occ = A.num_occ + B.num_occ;
    ABo.num_occ = ABn.num_occ;

    ABn.C_occ = ABn.C.leftCols(ABn.num_occ);
    ABo.C_occ = ABo.C.leftCols(ABo.num_occ);

    ABn.D= ABn.C_occ * ABn.C_occ.transpose();
    fmt::print("DABb\n{}\n", ABn.D);
    ABo.D= ABo.C_occ * ABo.C_occ.transpose();
    fmt::print("DABb\n{}\n", ABo.D);
    const auto kind_ab = kind_a;

    compute_energies<kind_ab>(ABn, hf_AB);
    compute_energies<kind_ab>(ABo, hf_AB);

    Energy E_ABn;
    E_ABn.coulomb = ABn.energy.coulomb - (A.energy.coulomb + B.energy.coulomb);
    E_ABn.exchange = ABn.energy.exchange - (A.energy.exchange + B.energy.exchange);
    E_ABn.core = ABn.energy.core - (A.energy.core + B.energy.core);
    double E_coul = E_ABn.coulomb + E_ABn.core + E_ABn.nuclear_repulsion;
    double E_tot = E_coul + E_ABn.exchange;

    double E_exch_rep = E_ABn.exchange + (ABo.energy.coulomb + ABo.energy.exchange + ABo.energy.core - ABn.energy.coulomb - ABn.energy.exchange - ABn.energy.core);

    fmt::print("E_coul: {}\n", E_coul * 2625.25);
    fmt::print("E_rep: {}\n", E_exch_rep * 2625.25);
}
