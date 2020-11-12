#include "logger.h"
#include "argparse.hpp"
#include "fchkreader.h"
#include <fmt/ostream.h>
#include "hf.h"
#include "wavefunction.h"
#include "element.h"
#include "util.h"
#include "gto.h"
#include "pairinteraction.h"
#include "disp.h"
#include "polarization.h"
#include <toml.hpp>

using tonto::qm::Wavefunction;
using tonto::qm::Energy;
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
using tonto::qm::BasisSet;

constexpr double kjmol_per_hartree{2625.46};


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


namespace impl
{
struct rotation
{
    Eigen::Matrix3d mat{Eigen::Matrix3d::Identity(3,3)};
};

struct translation
{
    Eigen::Vector3d vec{Eigen::Vector3d::Zero(3)};
};

}

namespace toml {
template<>
struct from<impl::rotation>
{
    static impl::rotation from_toml(const value& v)
    {
        impl::rotation rot;
        auto arr = toml::get<toml::array>(v);
        if(arr.size() == 9) {
            for(size_t i = 0; i < 9; i++) {
                double v = arr[i].is_floating() ? arr[i].as_floating(std::nothrow) :
                                              static_cast<double>(arr[i].as_integer());
                rot.mat(i / 3, i % 3) = v;
            }
        }
        else if(arr.size() == 3)
        {
            for(size_t i = 0; i < 3; i++){
                auto row = toml::get<toml::array>(arr[i]);
                for(size_t j = 0; j < 3; j++)
                {
                    double v = row[j].is_floating() ? row[j].as_floating(std::nothrow) :
                                                  static_cast<double>(row[j].as_integer());
                    rot.mat(i, j) = v;
                }
            }
        }
        else {
            std::cerr << toml::format_error(
            "[error] 3D rotation matrix has invalid length",
            v, "Expecting a [3, 3] or [9] array of int or double")
            << std::endl;
        }
        return rot;
    }
};

template<>
struct from<impl::translation>
{
    static impl::translation from_toml(const value& v)
    {
        impl::translation trans;
        auto arr = toml::get<toml::array>(v);
        if(arr.size() == 3)
        {
            for(size_t i = 0; i < 3; i++) {
                double v = arr[i].is_floating() ? arr[i].as_floating(std::nothrow) :
                                              static_cast<double>(arr[i].as_integer());
                trans.vec(i) = v;
            }
        }
        else
        {
            std::cerr << toml::format_error(
            "[error] 3D translation vector has invalid length",
            v, "Expecting a [3] array of int or double")
            << std::endl;
        }
        return trans;
    }
};
}

int main(int argc, const char **argv) {
    const auto input = toml::parse((argc > 1) ? argv[1] : "ce.toml");
    const auto pair_interaction_table = toml::find(input, "interaction");
    const auto global_settings_table = toml::find(input, "global");

    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();

    using tonto::parallel::nthreads;
    nthreads = toml::find_or<int>(global_settings_table, "threads", 1);
    omp_set_num_threads(nthreads);

    const std::string model_name = toml::find_or<std::string>(pair_interaction_table, "model", "ce-b3lyp");
    const std::string fchk_filename_a = toml::find_or<std::string>(pair_interaction_table, "monomer_a", "a.fchk");
    const std::string fchk_filename_b = toml::find_or<std::string>(pair_interaction_table, "monomer_b", "b.fchk");

    tonto::Mat3 rotation_a = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_a", impl::rotation{}).mat;
    tonto::Mat3 rotation_b = toml::find_or<impl::rotation>(pair_interaction_table, "rotation_b", impl::rotation{}).mat;
    fmt::print("Rotation of monomer A:\n{}\n", rotation_a);
    fmt::print("Rotation of monomer B:\n{}\n", rotation_b);

    tonto::Vec3 translation_a = toml::find_or<impl::translation>(pair_interaction_table, "translation_a", impl::translation{}).vec;
    tonto::Vec3 translation_b = toml::find_or<impl::translation>(pair_interaction_table, "translation_b", impl::translation{}).vec;
    fmt::print("Translation of monomer A:\n{}\n", translation_a);
    fmt::print("Translation of monomer B:\n{}\n", translation_b);

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
    A.apply_transformation(rotation_a, translation_a);

    fmt::print("Geometry after transformation ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_a, "sym", "x", "y", "z");
    for (const auto &atom : A.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }

    tonto::hf::HartreeFock hf_a(A.atoms, A.basis);

    Wavefunction B(fchk_b);
    B.apply_transformation(rotation_b, translation_b);
    tonto::log::info("Finished reading {}", fchk_filename_b);
    fmt::print("Geometry after transformation ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", fchk_filename_b, "sym", "x", "y", "z");
    for (const auto &atom : B.atoms) {
        fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                   atom.x, atom.y, atom.z);
    }
    tonto::hf::HartreeFock hf_b(B.atoms, B.basis);

    tonto::log::info("Computing monomer energies for {}", fchk_filename_a);

    compute_energies(A, hf_a);
    tonto::log::info("Finished monomer energies for {}", fchk_filename_a);

    tonto::log::info("Computing monomer energies for {}", fchk_filename_b);
    if(fchk_filename_a == fchk_filename_b) {
        tonto::log::info("Skipping computing monomer enegies for B: same source wavefunction as A");
        B.energy = A.energy;
    }
    else {
        compute_energies(B, hf_b);
    }
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

    fmt::print("\n\nFinal result\n\nE_coul           {: 12.6f}\n", E_coul * kjmol_per_hartree);
    fmt::print("E_rep            {: 12.6f}\n", E_XR * kjmol_per_hartree);

    auto e_pol = compute_polarization_energy(A, hf_a, B, hf_b);
    fmt::print("E_pol            {: 12.6f}\n", e_pol * kjmol_per_hartree);
    auto e_disp = tonto::disp::ce_model_dispersion_energy(A.atoms, B.atoms);
    fmt::print("E_disp           {: 12.6f}\n", e_disp * kjmol_per_hartree);

    if(model_name == "ce-b3lyp")
    {
        fmt::print("E_tot (CE-B3LYP) {: 12.6f}\n",
                    tonto::interaction::CE_B3LYP_631Gdp.scaled_total(E_coul, E_XR, e_pol, e_disp) * kjmol_per_hartree);
    }
    else if (model_name == "ce-hf")
    {
        fmt::print("E_tot (CE-HF)    {: 12.6f}\n",
                    tonto::interaction::CE_HF_321G.scaled_total(E_coul, E_XR, e_pol, e_disp) * kjmol_per_hartree);
    }
    else
    {
        fmt::print("E_tot (unscaled) {: 12.6f}\n", (E_coul + E_XR + e_pol + e_disp) * kjmol_per_hartree);
    }

}
