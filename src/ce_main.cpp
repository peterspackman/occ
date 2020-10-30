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

    auto basis_a = fchk_a.basis_set();
    auto basis_b = fchk_b.basis_set();
    MatRM CA = fchk_a.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(basis_a, CA);
    MatRM CA_occ = CA.leftCols(fchk_a.num_alpha());
    MatRM DA = CA_occ * CA_occ.transpose();
    MatRM CB = fchk_b.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(basis_b, CB);
    MatRM CB_occ = CB.leftCols(fchk_a.num_alpha());
    MatRM DB = CB_occ * CB_occ.transpose();
    tonto::log::info("Finished reading SCF density matrices");
    tonto::log::info("Matrices are the same: {}", all_close(DA, DB, 1e-05));
    tonto::hf::HartreeFock hf_a(fchk_a.atoms(), basis_a);
    tonto::hf::HartreeFock hf_b(fchk_b.atoms(), basis_b);

    MatRM JA, KA, JB, KB, VA, VB, TA, TB, HA, HB;
    VA = hf_a.compute_nuclear_attraction_matrix();
    double e_nuc_a = expectation<SpinorbitalKind::Restricted>(DA, VA);
    tonto::log::info("Computed nuclear attraction for A, energy = {}", e_nuc_a);
    TA = hf_a.compute_kinetic_matrix();
    double e_kinetic_a = expectation<SpinorbitalKind::Restricted>(DA, TA);
    tonto::log::info("Computed kinetic energy for A, energy = {}", e_kinetic_a);
    HA = VA + TA;
    double e_core_a = expectation<SpinorbitalKind::Restricted>(DA, HA);
    tonto::log::info("Computed core hamiltonian for A, energy = {}", e_core_a);

    VB = hf_a.compute_nuclear_attraction_matrix();
    double e_nuc_b = expectation<SpinorbitalKind::Restricted>(DB, VB);
    tonto::log::info("Computed nuclear attraction for B, energy = {}", e_nuc_b);
    TB = hf_a.compute_kinetic_matrix();
    double e_kinetic_b = expectation<SpinorbitalKind::Restricted>(DB, TB);
    tonto::log::info("Computed kinetic energy for B, energy = {}", e_kinetic_b);
    HB = VB + TB;
    double e_core_b = expectation<SpinorbitalKind::Restricted>(DB, HB);
    tonto::log::info("Computed core hamiltonian for B, energy = {}", e_core_b);

    std::tie(JA, KA) = hf_a.compute_JK(kind_a, DA);
    tonto::log::info("Computed JK matrices for molecule A");
    double coul_a = expectation<SpinorbitalKind::Restricted>(DA, JA);
    double exchange_a = expectation<SpinorbitalKind::Restricted>(DA, KA);
    tonto::log::info("Coulomb: {}, Exchange: {}", coul_a, exchange_a);

    std::tie(JB, KB) = hf_b.compute_JK(kind_b, DB);
    tonto::log::info("Computed JK matrices for molecule B");
    double coul_b = expectation<SpinorbitalKind::Restricted>(DB, JB);
    double exchange_b = expectation<SpinorbitalKind::Restricted>(DB, KB);
    tonto::log::info("Coulomb: {}, Exchange: {}", coul_b, exchange_b);

    tonto::Vec e_a = fchk_a.alpha_mo_energies();
    tonto::Vec e_b = fchk_b.alpha_mo_energies();
    MatRM CAB, e_ab;
    std::tie(CAB, e_ab) = merge_molecular_orbitals(CA, CB, e_a, e_b);
    fmt::print("MO A: {}\n{}\n", e_a.transpose(), CA);
    fmt::print("MO B: {}\n{}\n", e_b.transpose(), CB);
    fmt::print("MO AB: {}\n{}\n", e_ab.transpose(), CAB);
    auto atoms_AB = merge_atoms(fchk_a.atoms(), fchk_b.atoms());

    fmt::print("Merged geometry\n{:3s} {:^10s} {:^10s} {:^10s}\n", "sym", "x", "y", "z");
    for (const auto &atom : atoms_AB) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    auto basis_AB = merge_basis_sets(basis_a, basis_b);
    auto shell2atom = basis_AB.shell2atom(atoms_AB);
    for(const auto &atom: shell2atom) {
        fmt::print(" {}", atom);
    }
    fmt::print("\n");
//    for(const auto& shell: basis_AB) {
//        fmt::print("{}\n", shell);
//    }
    auto hf_AB = tonto::hf::HartreeFock(atoms_AB, basis_AB);
    MatRM S_AB = hf_AB.compute_overlap_matrix();
    fmt::print("S\n{}\n", S_AB);
    MatRM X_AB, X_inv_AB, DAB, CAB_occ;
    double X_AB_condition_number;
    double S_condition_number_threshold = 1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X_AB, X_inv_AB, X_AB_condition_number) = tonto::conditioning_orthogonalizer(S_AB, S_condition_number_threshold);
    fmt::print("Condition number: {}\n", X_AB_condition_number);

    fmt::print("X\n{}\n", X_AB);
    fmt::print("X_inv\n{}\n", X_inv_AB);
    CAB =  CAB * X_inv_AB;
    CAB_occ = CAB.leftCols(fchk_a.num_alpha() + fchk_b.num_alpha());
    DAB = CAB_occ * CAB_occ.transpose();
    fmt::print("DAB\n{}\n", DAB);
    MatRM JAB, KAB;
    auto kind_ab = kind_a;
    std::tie(JAB, KAB) = hf_AB.compute_JK(kind_ab, DAB);
    double coul_ab = expectation<SpinorbitalKind::Restricted>(DAB, JAB);
    double exchange_ab = expectation<SpinorbitalKind::Restricted>(DAB, KAB);
    tonto::log::info("Coulomb: {}, Exchange: {}", coul_ab, exchange_ab);

    MatRM VAB = hf_AB.compute_nuclear_attraction_matrix();
    double e_nuc_ab = expectation<SpinorbitalKind::Restricted>(DAB, VAB);
    tonto::log::info("Computed nuclear attraction for AB, energy = {}", e_nuc_ab);
    MatRM TAB = hf_AB.compute_kinetic_matrix();
    double e_kinetic_ab = expectation<SpinorbitalKind::Restricted>(DAB, TAB);
    tonto::log::info("Computed kinetic energy for AB, energy = {}", e_kinetic_ab);
    MatRM HAB = VAB + TAB;
    double e_core_ab = expectation<SpinorbitalKind::Restricted>(DAB, HAB);
    tonto::log::info("Computed core hamiltonian for B, energy = {}", e_core_ab);
    fmt::print("Differences:\nEcoul = {}\nEexch = {}\nEnuc  = {}\nEkin  = {}\nEcore = {}\n",
               coul_ab - coul_a - coul_b, exchange_ab - exchange_a - exchange_b,
               e_nuc_ab - e_nuc_a - e_nuc_b, e_kinetic_ab - e_kinetic_a - e_kinetic_b,
               e_core_ab - e_core_a - e_core_b);

}
