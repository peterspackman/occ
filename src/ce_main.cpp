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
#include "disp.h"

using tonto::io::FchkReader;
using tonto::hf::HartreeFock;
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

constexpr double kjmol_per_hartree{2625.46};

const std::array<double, 110> Thakkar_atomic_polarizability{
     4.50,   1.38, 164.04,  37.74,  20.43,  11.67,   7.26,   5.24,   3.70,   2.66, 
   162.88,  71.22,  57.79,  37.17,  24.93,  19.37,  14.57,  11.09, 291.10, 157.90, 
   142.30, 114.30,  97.30,  94.70,  75.50,  63.90,  57.70,  51.10,  45.50,  38.35, 
    52.91,  40.80,  29.80,  26.24,  21.13,  16.80, 316.20, 199.00, 153.00, 121.00, 
   106.00,  86.00,  77.00,  65.00,  58.00,  32.00,  52.46,  47.55,  68.67,  57.30, 
    42.20,  38.10,  32.98,  27.06, 396.00, 273.50, 210.00, 200.00, 190.00, 212.00, 
   203.00, 194.00, 187.00, 159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30, 
   148.00, 109.00,  88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73, 
    71.72,  60.05,  48.60,  43.62,  40.73,  33.18, 315.20, 246.20, 217.00, 217.00, 
   171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00, 161.00, 
   123.00, 118.00,   0.00,   0.00,   0.00,   0.00, 0.00,   0.00,   0.00,   0.00
};

// Atomic polarizibilities for charged species
// if not assigned, these should be the same as the uncharged
// +/- are in the same table, double charges are implicit e.g. for Ca
// + will be SIGNIFICANTLY smaller than neutral, - should be a bit larger than neutral
// val for iodine was interpolated

const std::array<double, 110> Charged_atomic_polarizibility{
     4.50,   1.38,  0.19,   0.052,  20.43,  11.67,   7.26,   5.24,   7.25,   2.66,
   0.986,   0.482,  57.79,  37.17,  24.93,  19.37,  21.20,  11.09,   5.40,   3.20,
   142.30, 114.30,  97.30,  94.70,  75.50,  63.90,  57.70,  51.10,  45.50,  38.35,
    52.91,  40.80,  29.80,  26.24,  27.90,  16.80,   9.10,   5.80, 153.00, 121.00,
   106.00,  86.00,  77.00,  65.00,  58.00,  32.00,  52.46,  47.55,  68.67,  57.30,
    42.20,  38.10,  39.60,  27.06,  15.70,  10.60, 210.00, 200.00, 190.00, 212.00,
   203.00, 194.00, 187.00, 159.00, 172.00, 165.00, 159.00, 153.00, 147.00, 145.30,
   148.00, 109.00,  88.00,  75.00,  65.00,  57.00,  51.00,  44.00,  36.06,  34.73,
    71.72,  60.05,  48.60,  43.62,  40.73,  33.18,  20.40,  13.40, 217.00, 217.00,
   171.00, 153.00, 167.00, 165.00, 157.00, 155.00, 153.00, 138.00, 133.00, 161.00,
   123.00, 118.00,   0.00,   0.00,   0.00,   0.00, 0.00,   0.00,   0.00,   0.00
};


struct Energy {
    double coulomb{0};
    double exchange{0};
    double nuclear_repulsion{0};
    double nuclear_attraction{0};
    double kinetic{0};
    double core{0};
    void print() const {
        constexpr auto format_string = "{:<10s} {:10.6f}\n";
        fmt::print(format_string, "E_coul", coulomb);
        fmt::print(format_string, "E_ex", exchange);
        fmt::print(format_string, "E_nn", nuclear_repulsion);
        fmt::print(format_string, "E_en", nuclear_attraction);
        fmt::print(format_string, "E_kin", kinetic);
        fmt::print(format_string, "E_1e", core);
    }
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
    wfn.V = hf.compute_nuclear_attraction_matrix();
    wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.D, wfn.V);
    wfn.T = hf.compute_kinetic_matrix();
    wfn.energy.kinetic = 2 * expectation<kind>(wfn.D, wfn.T);
    wfn.H = wfn.V + wfn.T;
    wfn.energy.core = 2 * expectation<kind>(wfn.D, wfn.H);
    std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.D);
    wfn.energy.coulomb = expectation<kind>(wfn.D, wfn.J);
    wfn.energy.exchange = - expectation<kind>(wfn.D, wfn.K);
    wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
}


MatRM symmetrically_orthonormalize(const MatRM& mat, const MatRM& metric)
{
    MatRM X, X_invT;
    size_t n_cond;
    double x_condition_number, condition_number;
    double threshold = 1.0 / std::numeric_limits<double>::epsilon();
    MatRM SS = mat.transpose() * metric * mat;
    std::tie(X, X_invT, n_cond, x_condition_number, condition_number) = tonto::gensqrtinv(SS, true, threshold);
    return mat * X;
}

MatRM symmorthonormalize_molecular_orbitals(const MatRM& mos, const MatRM& overlap, size_t n_occ)
{
    MatRM result(mos.rows(), mos.cols());
    size_t n_virt = mos.cols() - n_occ;
    MatRM C_occ = mos.leftCols(n_occ);
    MatRM C_virt = mos.rightCols(n_virt);
    result.leftCols(n_occ) = symmetrically_orthonormalize(C_occ, overlap);
    result.rightCols(n_virt) = symmetrically_orthonormalize(C_virt, overlap);
    return result;
}

double compute_polarization_energy(const Wavefunction &wfn_a, const HartreeFock &hf_a,
                                   const Wavefunction &wfn_b, const HartreeFock &hf_b)
{
    tonto::Mat3N apos(3, wfn_a.atoms.size());
    for(size_t i = 0; i < wfn_a.atoms.size(); i++) {
        apos(0, i) = wfn_a.atoms[i].x;
        apos(1, i) = wfn_a.atoms[i].y;
        apos(2, i) = wfn_a.atoms[i].z;
    }
    tonto::Mat3N bpos(3, wfn_b.atoms.size());
    for(size_t i = 0; i < wfn_b.atoms.size(); i++){
        bpos(0, i) = wfn_b.atoms[i].x;
        bpos(1, i) = wfn_b.atoms[i].y;
        bpos(2, i) = wfn_b.atoms[i].z;
    }

    // fields (incl. sign) have been checked and agree with both finite difference
    // method and tonto
    tonto::Mat3N field_a = hf_b.electronic_electric_field_contribution(wfn_b.D, apos);
    field_a += hf_b.nuclear_electric_field_contribution(apos);
    tonto::Mat3N field_b = hf_a.electronic_electric_field_contribution(wfn_a.D, bpos);
    field_b += hf_a.nuclear_electric_field_contribution(bpos);

    auto fsq_a = field_a.colwise().squaredNorm();
    auto fsq_b = field_b.colwise().squaredNorm();
    fmt::print("F_sq_A\n{}\n", fsq_a);
    double epol = 0.0;
    for(size_t i = 0; i < wfn_a.atoms.size(); i++)
    {
        size_t n = wfn_a.atoms[i].atomic_number;
        double pol = Thakkar_atomic_polarizability[n - 1];
        epol += pol * fsq_a(i);
    }
    fmt::print("F_sq_B\n{}\n", fsq_a);
    for(size_t i = 0; i < wfn_b.atoms.size(); i++)
    {
        size_t n = wfn_b.atoms[i].atomic_number;
        double pol = Thakkar_atomic_polarizability[n - 1];
        epol += pol * fsq_b(i);
    }
    return epol * -0.5;
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

    A.C = fchk_a.alpha_mo_coefficients();
    A.C = fchk_a.reordered_mo_coefficients_from_gaussian_convention(A.basis, A.C);
    A.C_occ = A.C.leftCols(A.num_occ);
    A.D = A.C_occ * A.C_occ.transpose();

    B.C = fchk_b.alpha_mo_coefficients();
    B.C = fchk_b.reordered_mo_coefficients_from_gaussian_convention(B.basis, B.C);
    B.C_occ = B.C.leftCols(B.num_occ);
    B.D = B.C_occ * B.C_occ.transpose();

    tonto::log::info("Finished reading SCF density matrices");
    tonto::hf::HartreeFock hf_a(A.atoms, A.basis);
    tonto::hf::HartreeFock hf_b(B.atoms, B.basis);

    A.mo_energies = fchk_a.alpha_mo_energies();
    B.mo_energies = fchk_b.alpha_mo_energies();

    compute_energies<kind_a>(A, hf_a);
    compute_energies<kind_b>(B, hf_b);

    ABn.C = MatRM(A.C.rows() + B.C.rows(), A.C.cols() + B.C.cols());
    ABn.mo_energies = tonto::Vec(A.mo_energies.rows() + B.mo_energies.rows());

    MatRM C_merged;
    tonto::Vec energies_merged;
    ABn.num_occ = A.num_occ + B.num_occ;

    tonto::log::info("Merging occupied orbitals, sorted by energy");
    // merge occupied orbitals, merging occupied separately from virtual
    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
        A.C.leftCols(A.num_occ), B.C.leftCols(B.num_occ),
        A.mo_energies.topRows(A.num_occ), B.mo_energies.topRows(B.num_occ));


    ABn.C.leftCols(ABn.num_occ) = C_merged;
    ABn.mo_energies.topRows(ABn.num_occ) = energies_merged;

    size_t nv_a = A.C.rows() - A.num_occ, nv_b = B.C.rows() - B.num_occ;
    size_t nv_ab = nv_a + nv_b;

    tonto::log::info("Merging virtual orbitals, sorted by energy");
    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
        A.C.rightCols(nv_a), B.C.leftCols(nv_b),
        A.mo_energies.bottomRows(nv_a), B.mo_energies.topRows(nv_b));
    ABn.C.rightCols(nv_ab) = C_merged;
    ABn.mo_energies.bottomRows(nv_ab) = energies_merged;

    ABn.atoms = merge_atoms(A.atoms, B.atoms);
    ABo.atoms = ABn.atoms;

    fmt::print("Merged geometry\n{:3s} {:^10s} {:^10s} {:^10s}\n", "sym", "x", "y", "z");
    for (const auto &atom : ABn.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    ABn.basis = merge_basis_sets(A.basis, B.basis);
    auto hf_AB = tonto::hf::HartreeFock(ABn.atoms, ABn.basis);


    tonto::log::info("Computing overlap matrix for merged orbitals");
    MatRM S_AB = hf_AB.compute_overlap_matrix();
    tonto::log::info("Orthonormalizing merged orbitals using overlap matrix");
    ABo.C = symmorthonormalize_molecular_orbitals(ABn.C, S_AB, ABn.num_occ);
    ABo.num_occ = ABn.num_occ;

    ABn.C_occ = ABn.C.leftCols(ABn.num_occ);
    ABo.C_occ = ABo.C.leftCols(ABo.num_occ);

    ABn.D= ABn.C_occ * ABn.C_occ.transpose();
    ABo.D= ABo.C_occ * ABo.C_occ.transpose();
    const auto kind_ab = kind_a;

    tonto::log::info("Computing non-orthogonal merged energies");
    compute_energies<kind_ab>(ABn, hf_AB);
    tonto::log::info("Computing orthogonal merged energies");
    compute_energies<kind_ab>(ABo, hf_AB);


    Energy E_ABn;
    E_ABn.kinetic = ABn.energy.kinetic - (A.energy.kinetic + B.energy.kinetic);
    E_ABn.coulomb = ABn.energy.coulomb - (A.energy.coulomb + B.energy.coulomb);
    E_ABn.exchange = ABn.energy.exchange - (A.energy.exchange + B.energy.exchange);
    E_ABn.core = ABn.energy.core - (A.energy.core + B.energy.core);
    E_ABn.nuclear_attraction = ABn.energy.nuclear_attraction - (A.energy.nuclear_attraction + B.energy.nuclear_attraction);
    E_ABn.nuclear_repulsion = ABn.energy.nuclear_repulsion - (A.energy.nuclear_repulsion + B.energy.nuclear_repulsion);
    double E_coul = E_ABn.coulomb + E_ABn.nuclear_attraction + E_ABn.nuclear_repulsion;
    double eABn = ABn.energy.core + ABn.energy.exchange + ABn.energy.coulomb;
    double eABo = ABo.energy.core + ABo.energy.exchange + ABo.energy.coulomb;
    double E_rep = eABo - eABn;
    double E_XR = E_ABn.exchange + E_rep;

    fmt::print("ABn\n");
    ABn.energy.print();
    fmt::print("ABo\n");
    ABo.energy.print();

    fmt::print("Results\n\nE_coul  {: 12.6f}\n", E_coul * kjmol_per_hartree);
    fmt::print("E_rep   {: 12.6f}\n", E_XR * kjmol_per_hartree);

    auto e_pol = compute_polarization_energy(A, hf_a, B, hf_b);
    fmt::print("E_pol   {: 12.6f}\n", e_pol * kjmol_per_hartree);
    auto e_disp = tonto::disp::d2_interaction_energy(A.atoms, B.atoms);
    fmt::print("E_disp  {: 12.6f}\n", e_disp * kjmol_per_hartree);

    fmt::print("E_tot (CE-B3LYP) {: 12.6f}\n", 
                tonto::interaction::CE_B3LYP_631Gdp.scaled_total(E_coul, E_XR, e_pol, e_disp) * kjmol_per_hartree);
}
