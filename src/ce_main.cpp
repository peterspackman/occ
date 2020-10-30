#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>
#include "hf.h"
#include "spinorbital.h"
#include "element.h"
#include "util.h"
#include "gto.h"

using tonto::io::FchkReader;
using tonto::qm::SpinorbitalKind;
using tonto::qm::expectation;
using tonto::chem::Element;
using tonto::util::all_close;
using tonto::util::join;
using tonto::gto::shell_component_labels;
using tonto::MatRM;
using tonto::Vec;

std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM& mo_a, const MatRM& mo_b, const Vec e_a, const Vec e_b)
{
    MatRM merged = MatRM::Zero(mo_a.rows() + mo_b.rows(), mo_a.cols() + mo_b.cols());
    Vec merged_energies(e_a.rows() + e_b.rows());
    merged_energies.topRows(e_a.rows()) = e_a;
    merged_energies.bottomRows(e_b.rows()) = e_b;
    std::vector<Eigen::Index> idxs;
    idxs.reserve(merged_energies.rows());
    for(Eigen::Index i = 0; i < merged_energies.rows(); i++) idxs.push_back(i);
    std::sort(idxs.begin(), idxs.end(), [&merged_energies](Eigen::Index a, Eigen::Index b) { return merged_energies(a) < merged_energies(b); });
    Vec sorted_energies(merged_energies.rows());
    for(Eigen::Index i = 0; i < merged_energies.rows(); i++) {
        Eigen::Index c = idxs[i];
        sorted_energies(i) = merged_energies(c);
        if(c >= mo_a.cols()) {
            merged.col(i).bottomRows(mo_b.rows()) = mo_b.col(c - mo_a.cols());
        }
        else {
            merged.col(i).topRows(mo_a.rows()) = mo_a.col(c);
        }
    }
    return {merged, sorted_energies};
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
    for(const auto&shell : fchk_a.basis_set()) {
        fmt::print("\n{}\n", shell);
    }
    FchkReader fchk_b(fchk_filename_b);
    tonto::log::info("Parsed fchk file: {}", fchk_filename_b);
    fmt::print("Input geometry ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : fchk_b.atoms()) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    for(const auto&shell : fchk_b.basis_set()) {
        fmt::print("\n{}\n", shell);
        fmt::print("Components: {}\n", join(shell_component_labels(shell.contr[0].l), " "));
    }

    auto basis_a = fchk_a.basis_set();
    auto basis_b = fchk_b.basis_set();
    tonto::MatRM CA = fchk_a.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(basis_a, CA);
    tonto::MatRM CA_occ = CA.leftCols(fchk_a.num_alpha());
    tonto::MatRM DA = CA_occ * CA_occ.transpose();
    tonto::MatRM CB = fchk_b.alpha_mo_coefficients();
    FchkReader::reorder_mo_coefficients_from_gaussian_convention(basis_b, CB);
    tonto::MatRM CB_occ = CB.leftCols(fchk_a.num_alpha());
    tonto::MatRM DB = CB_occ * CB_occ.transpose();
    tonto::log::info("Finished reading SCF density matrices");
    tonto::log::info("Matrices are the same: {}", all_close(DA, DB, 1e-05));
    tonto::hf::HartreeFock hf_a(fchk_a.atoms(), basis_a);
    tonto::hf::HartreeFock hf_b(fchk_b.atoms(), basis_b);

    tonto::MatRM JA, KA, JB, KB, VA, VB, TA, TB, HA, HB;
    VA = hf_a.compute_nuclear_attraction_matrix();
    tonto::log::info("Computed nuclear attraction for A, energy = {}", expectation<SpinorbitalKind::Restricted>(DA, VA));
    TA = hf_a.compute_kinetic_matrix();
    tonto::log::info("Computed kinetic energy for A, energy = {}", expectation<SpinorbitalKind::Restricted>(DA, TA));
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
    tonto::MatRM fock_A = hf_a.compute_fock(kind_a, DA);
    std::tie(JB, KB) = hf_b.compute_JK(kind_b, DB);
    tonto::log::info("Computed JK matrices for molecule B");
    tonto::log::info("Coulomb: {}, Exchange: {}",
        expectation<SpinorbitalKind::Restricted>(DB, JB),
        expectation<SpinorbitalKind::Restricted>(DB, KB)
    );
    tonto::MatRM fock_B = hf_b.compute_fock(kind_b, DB);
    tonto::Vec e_a = fchk_a.alpha_mo_energies();
    tonto::Vec e_b = fchk_b.alpha_mo_energies();
    tonto::MatRM CAB, e_ab;
    std::tie(CAB, e_ab) = merge_molecular_orbitals(CA, CB, e_a, e_b);
    fmt::print("MO A energies:\n{}\nCoeffs:\n{}\n", e_a, CA);
    fmt::print("MO B energies:\n{}\nCoeffs:\n{}\n", e_b, CB);
    fmt::print("MO AB energies:\n{}\nCoeffs:\n{}\n", e_ab, CAB);

}
