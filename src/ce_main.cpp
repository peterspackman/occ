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
#include "polarization.h"

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
MatRM symmorthonormalize_molecular_orbitals(const MatRM& mos, const MatRM& overlap, size_t n_occ);

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
    Wavefunction() {}

    Wavefunction(const FchkReader& fchk) :
        spinorbital_kind(fchk.spinorbital_kind()),
        num_alpha(fchk.num_alpha()),
        num_beta(fchk.num_beta()),
        num_electrons(fchk.num_electrons()),
        basis(fchk.basis_set()),
        nbf(tonto::qm::nbf(basis)),
        atoms(fchk.atoms())
    {
        set_molecular_orbitals(fchk);
        compute_density_matrix();
    }


    Wavefunction(const Wavefunction &wfn_a, const Wavefunction &wfn_b) :
        num_alpha(wfn_a.num_alpha + wfn_b.num_alpha),
        num_beta(wfn_a.num_beta + wfn_b.num_beta),
        basis(merge_basis_sets(wfn_a.basis, wfn_b.basis)),
        nbf(wfn_a.nbf + wfn_b.nbf),
        atoms(merge_atoms(wfn_a.atoms, wfn_b.atoms))
    {
        spinorbital_kind = (wfn_a.is_restricted() && wfn_b.is_restricted()) ? SpinorbitalKind::Restricted : SpinorbitalKind::Unrestricted;

        size_t rows, cols;
        if(is_restricted()) std::tie(rows, cols) = tonto::qm::matrix_dimensions<SpinorbitalKind::Restricted>(nbf);
        else std::tie(rows, cols) = tonto::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
        C = MatRM(rows, cols);
        mo_energies = tonto::Vec(rows);
        // temporaries for merging orbitals
        MatRM C_merged;
        tonto::Vec energies_merged;
        tonto::log::debug("Merging occupied orbitals, sorted by energy");
        if(wfn_a.is_restricted() && wfn_b.is_restricted())
        {
            // merge occupied orbitals
            std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                wfn_a.C.leftCols(wfn_a.num_alpha), wfn_b.C.leftCols(wfn_b.num_alpha),
                wfn_a.mo_energies.topRows(wfn_a.num_alpha), wfn_b.mo_energies.topRows(wfn_b.num_alpha)
            );
            C.leftCols(num_alpha) = C_merged;
            mo_energies.topRows(num_alpha) = energies_merged;

            // merge virtual orbitals
            size_t nv_a = wfn_a.C.rows() - wfn_a.num_alpha, nv_b = wfn_b.C.rows() - wfn_b.num_alpha;
            size_t nv_ab = nv_a + nv_b;

            tonto::log::info("Merging virtual orbitals, sorted by energy");
            std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                wfn_a.C.rightCols(nv_a), wfn_b.C.rightCols(nv_b),
                wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
            C.rightCols(nv_ab) = C_merged;
            mo_energies.bottomRows(nv_ab) = energies_merged;
        }
        else {
            if(wfn_a.is_restricted()) {
                { //alpha
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.leftCols(wfn_a.num_alpha), wfn_b.C.alpha().leftCols(wfn_b.num_alpha),
                        wfn_a.mo_energies.topRows(wfn_a.num_alpha), wfn_b.mo_energies.alpha().topRows(wfn_b.num_alpha)
                    );
                    C.alpha().leftCols(num_alpha) = C_merged;
                    mo_energies.alpha().topRows(num_alpha) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.rows() - wfn_a.num_alpha, nv_b = wfn_b.C.alpha().rows() - wfn_b.num_alpha;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.rightCols(nv_a), wfn_b.C.alpha().rightCols(nv_b),
                        wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.alpha().bottomRows(nv_b));
                    C.alpha().rightCols(nv_ab) = C_merged;
                    mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
                }
                { //beta
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.leftCols(wfn_a.num_beta), wfn_b.C.beta().leftCols(wfn_b.num_beta),
                        wfn_a.mo_energies.topRows(wfn_a.num_beta), wfn_b.mo_energies.beta().topRows(wfn_b.num_beta)
                    );
                    C.beta().leftCols(num_beta) = C_merged;
                    mo_energies.beta().topRows(num_beta) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.rows() - wfn_a.num_beta, nv_b = wfn_b.C.beta().rows() - wfn_b.num_beta;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.rightCols(nv_a), wfn_b.C.beta().rightCols(nv_b),
                        wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.beta().bottomRows(nv_b));
                    C.beta().rightCols(nv_ab) = C_merged;
                    mo_energies.beta().bottomRows(nv_ab) = energies_merged;
                }
            }
            else if(wfn_b.is_restricted()) {
                { //alpha
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.alpha().leftCols(wfn_a.num_alpha), wfn_b.C.leftCols(wfn_b.num_alpha),
                        wfn_a.mo_energies.alpha().topRows(wfn_a.num_alpha), wfn_b.mo_energies.topRows(wfn_b.num_alpha)
                    );
                    C.alpha().leftCols(num_alpha) = C_merged;
                    mo_energies.alpha().topRows(num_alpha) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.alpha().rows() - wfn_a.num_alpha, nv_b = wfn_b.C.rows() - wfn_b.num_alpha;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.alpha().rightCols(nv_a), wfn_b.C.rightCols(nv_b),
                        wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
                    C.alpha().rightCols(nv_ab) = C_merged;
                    mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
                }
                { //beta
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.beta().leftCols(wfn_a.num_beta), wfn_b.C.leftCols(wfn_b.num_beta),
                        wfn_a.mo_energies.beta().topRows(wfn_a.num_beta), wfn_b.mo_energies.topRows(wfn_b.num_beta)
                    );
                    C.beta().leftCols(num_beta) = C_merged;
                    mo_energies.beta().topRows(num_beta) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.beta().rows() - wfn_a.num_beta, nv_b = wfn_b.C.rows() - wfn_b.num_beta;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.beta().rightCols(nv_a), wfn_b.C.rightCols(nv_b),
                        wfn_a.mo_energies.beta().bottomRows(nv_a), wfn_b.mo_energies.bottomRows(nv_b));
                    C.beta().rightCols(nv_ab) = C_merged;
                    mo_energies.beta().bottomRows(nv_ab) = energies_merged;
                }
            }
            else {
                { //alpha
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.alpha().leftCols(wfn_a.num_alpha), wfn_b.C.alpha().leftCols(wfn_b.num_alpha),
                        wfn_a.mo_energies.alpha().topRows(wfn_a.num_alpha), wfn_b.mo_energies.alpha().topRows(wfn_b.num_alpha)
                    );
                    C.alpha().leftCols(num_alpha) = C_merged;
                    mo_energies.alpha().topRows(num_alpha) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.alpha().rows() - wfn_a.num_alpha, nv_b = wfn_b.C.alpha().rows() - wfn_b.num_alpha;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.alpha().rightCols(nv_a), wfn_b.C.alpha().rightCols(nv_b),
                        wfn_a.mo_energies.bottomRows(nv_a), wfn_b.mo_energies.alpha().bottomRows(nv_b));
                    C.alpha().rightCols(nv_ab) = C_merged;
                    mo_energies.alpha().bottomRows(nv_ab) = energies_merged;
                }
                { //beta
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.beta().leftCols(wfn_a.num_beta), wfn_b.C.beta().leftCols(wfn_b.num_beta),
                        wfn_a.mo_energies.beta().topRows(wfn_a.num_beta), wfn_b.mo_energies.beta().topRows(wfn_b.num_beta)
                    );
                    C.beta().leftCols(num_beta) = C_merged;
                    mo_energies.beta().topRows(num_beta) = energies_merged;

                    // merge virtual orbitals
                    size_t nv_a = wfn_a.C.beta().rows() - wfn_a.num_beta, nv_b = wfn_b.C.beta().rows() - wfn_b.num_beta;
                    size_t nv_ab = nv_a + nv_b;

                    tonto::log::info("Merging virtual orbitals, sorted by energy");
                    std::tie(C_merged, energies_merged) = merge_molecular_orbitals(
                        wfn_a.C.beta().rightCols(nv_a), wfn_b.C.beta().rightCols(nv_b),
                        wfn_a.mo_energies.beta().bottomRows(nv_a), wfn_b.mo_energies.beta().bottomRows(nv_b));
                    C.beta().rightCols(nv_ab) = C_merged;
                    mo_energies.beta().bottomRows(nv_ab) = energies_merged;
                }
            }
        }
        set_occupied_orbitals();
    }

    bool is_restricted() const { return spinorbital_kind == SpinorbitalKind::Restricted; }
    SpinorbitalKind spinorbital_kind{SpinorbitalKind::Restricted};
    int num_alpha;
    int num_beta;
    int num_electrons;
    BasisSet basis;
    size_t nbf{0};
    std::vector<libint2::Atom> atoms;

    size_t n_alpha() const { return num_alpha; }
    size_t n_beta() const { return num_beta; }
    MatRM C, C_occ, D, T, V, H, J, K;
    Vec mo_energies;
    Energy energy;

    void set_occupied_orbitals()
    {
        if(C.size() == 0) { return; }
        if(spinorbital_kind == SpinorbitalKind::General) {
            throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
        }
        else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            C_occ = MatRM::Zero(2 * nbf, std::max(num_alpha, num_beta));
            C_occ.block(0, 0, nbf, num_alpha) = C.alpha().leftCols(num_alpha);
            C_occ.block(nbf, 0, nbf, num_beta) = C.beta().leftCols(num_beta);
        }
        else {
            C_occ = C.leftCols(num_alpha);
        }
    }

    void apply_rotation(const tonto::Mat3& rot)
    {

    }

    void set_molecular_orbitals(const FchkReader& fchk)
    {
        size_t rows, cols;
        nbf = tonto::qm::nbf(basis);

        if(spinorbital_kind == SpinorbitalKind::General) {
            throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
        }
        else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            std::tie(rows, cols) = tonto::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
            C = MatRM(rows, cols);
            mo_energies = Vec(rows);
            C.alpha() = fchk.alpha_mo_coefficients();
            C.beta() = fchk.beta_mo_coefficients();
            mo_energies.alpha() = fchk.alpha_mo_energies();
            mo_energies.beta() = fchk.beta_mo_energies();
            C.alpha() = fchk.convert_mo_coefficients_from_g09_convention(basis, C.alpha());
            C.beta() = fchk.convert_mo_coefficients_from_g09_convention(basis, C.beta());
        }
        else {
            C = fchk.alpha_mo_coefficients();
            mo_energies = fchk.alpha_mo_energies();
            C = fchk.convert_mo_coefficients_from_g09_convention(basis, C);
        }
        set_occupied_orbitals();
    }

    void compute_density_matrix() {
        if(spinorbital_kind == SpinorbitalKind::General) {
            throw std::runtime_error("Reading MOs from g09 unsupported for General spinorbitals");
        }
        else if(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            size_t rows, cols;
            std::tie(rows, cols) = tonto::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
            D = tonto::MatRM(rows, cols);
            D.alpha() = C_occ.block(0, 0, nbf, num_alpha) * C_occ.block(0, 0, nbf, num_alpha).transpose();
            D.beta() = C_occ.block(nbf, 0, nbf, num_beta) * C_occ.block(nbf, 0, nbf, num_beta).transpose();
            D *= 0.5;
        }
        else {
            D = C_occ * C_occ.transpose();
        }
    }

    void symmetric_orthonormalize_molecular_orbitals(const MatRM& overlap)
    {
        if(spinorbital_kind == SpinorbitalKind::Restricted)
        {
            C = symmorthonormalize_molecular_orbitals(C, overlap, num_alpha);
        }
        else {
            C.alpha() = symmorthonormalize_molecular_orbitals(C.alpha(), overlap, num_alpha);
            C.beta() = symmorthonormalize_molecular_orbitals(C.beta(), overlap, num_beta);
        }
        set_occupied_orbitals();
    }
};

template<SpinorbitalKind kind>
void compute_energies(Wavefunction& wfn, tonto::hf::HartreeFock& hf)
{
    if constexpr(kind == SpinorbitalKind::Restricted) {
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
    else {
        size_t rows, cols;
        std::tie(rows, cols) = tonto::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(wfn.nbf);
        wfn.T = MatRM(rows, cols);
        wfn.V = MatRM(rows, cols);
        wfn.T.alpha() = hf.compute_kinetic_matrix();
        wfn.T.beta() = wfn.T.alpha();
        wfn.V.alpha() = hf.compute_nuclear_attraction_matrix();
        wfn.V.beta() = wfn.V.alpha();
        wfn.H = wfn.V + wfn.T;
        wfn.energy.nuclear_attraction = 2 * expectation<kind>(wfn.D, wfn.V);
        wfn.energy.kinetic = 2 * expectation<kind>(wfn.D, wfn.T);
        wfn.energy.core = 2 * expectation<kind>(wfn.D, wfn.H);
        std::tie(wfn.J, wfn.K) = hf.compute_JK(kind, wfn.D);
        wfn.energy.coulomb = expectation<kind>(wfn.D, wfn.J);
        wfn.energy.exchange = - expectation<kind>(wfn.D, wfn.K);
        wfn.energy.nuclear_repulsion = hf.nuclear_repulsion_energy();
    }

}


void compute_energies(Wavefunction &wfn, tonto::hf::HartreeFock &hf)
{
    if(wfn.is_restricted()) return compute_energies<SpinorbitalKind::Restricted>(wfn, hf);
    else return compute_energies<SpinorbitalKind::Unrestricted>(wfn, hf);
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
    tonto::IVec anums(wfn_a.atoms.size());
    for(size_t i = 0; i < wfn_a.atoms.size(); i++) {
        apos(0, i) = wfn_a.atoms[i].x;
        apos(1, i) = wfn_a.atoms[i].y;
        apos(2, i) = wfn_a.atoms[i].z;
        anums(i) = wfn_a.atoms[i].atomic_number;
    }
    tonto::Mat3N bpos(3, wfn_b.atoms.size());
    tonto::IVec bnums(wfn_b.atoms.size());
    for(size_t i = 0; i < wfn_b.atoms.size(); i++){
        bpos(0, i) = wfn_b.atoms[i].x;
        bpos(1, i) = wfn_b.atoms[i].y;
        bpos(2, i) = wfn_b.atoms[i].z;
        bnums(i) = wfn_b.atoms[i].atomic_number;
    }

    // fields (incl. sign) have been checked and agree with both finite difference
    // method and tonto
    tonto::Mat3N field_a = hf_b.electronic_electric_field_contribution(wfn_b.D, apos);
    field_a += hf_b.nuclear_electric_field_contribution(apos);
    tonto::Mat3N field_b = hf_a.electronic_electric_field_contribution(wfn_a.D, bpos);
    field_b += hf_a.nuclear_electric_field_contribution(bpos);

    using tonto::pol::ce_model_polarization_energy;
    double e_pol = ce_model_polarization_energy(anums, field_a) +
                   ce_model_polarization_energy(bnums, field_b);
    return e_pol;
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

    Wavefunction A(fchk_a);
    tonto::log::info("Finished reading {}", fchk_filename_a);
    tonto::hf::HartreeFock hf_a(A.atoms, A.basis);

    Wavefunction B(fchk_b);
    tonto::log::info("Finished reading {}", fchk_filename_b);
    tonto::hf::HartreeFock hf_b(B.atoms, B.basis);

    tonto::log::info("Computing monomer energies for {}", fchk_filename_a);

    compute_energies(A, hf_a);
    tonto::log::info("Finished monomer energies for {}", fchk_filename_a);

    tonto::log::info("Computing monomer energies for {}", fchk_filename_b);
    compute_energies(B, hf_b);
    tonto::log::info("Finished monomer energies for {}", fchk_filename_b);

    fmt::print("{} energies\n", fchk_filename_a);
    A.energy.print();
    fmt::print("{} energies\n", fchk_filename_b);
    B.energy.print();


    Wavefunction ABn(A, B);
    fmt::print("Merged geometry\n{:3s} {:^10s} {:^10s} {:^10s}\n", "sym", "x", "y", "z");
    for (const auto &atom : ABn.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    auto hf_AB = tonto::hf::HartreeFock(ABn.atoms, ABn.basis);

    Wavefunction ABo = ABn;
    tonto::log::info("Computing overlap matrix for merged orbitals");
    MatRM S_AB = hf_AB.compute_overlap_matrix();
    tonto::log::info("Orthonormalizing merged orbitals using overlap matrix");
    ABo.symmetric_orthonormalize_molecular_orbitals(S_AB);

    ABn.compute_density_matrix();
    ABo.compute_density_matrix();

    tonto::log::info("Computing non-orthogonal merged energies");
    compute_energies(ABn, hf_AB);
    tonto::log::info("Computing orthogonal merged energies");
    compute_energies(ABo, hf_AB);


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
    auto e_disp = tonto::disp::ce_model_dispersion_energy(A.atoms, B.atoms);
    fmt::print("E_disp  {: 12.6f}\n", e_disp * kjmol_per_hartree);

    fmt::print("E_tot (CE-B3LYP) {: 12.6f}\n", 
                tonto::interaction::CE_B3LYP_631Gdp.scaled_total(E_coul, E_XR, e_pol, e_disp) * kjmol_per_hartree);
}
