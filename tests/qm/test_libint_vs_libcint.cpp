#include "catch.hpp"
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/parallel.h>
#include <occ/io/json_basis.h>
#include <occ/qm/basisset.h>
#include <occ/qm/fock.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <vector>

using occ::qm::cint::Operator;
using Kind = occ::qm::OccShell::Kind;
using occ::Mat;
using occ::qm::IntegralEngine;

IntegralEngine
read_atomic_orbital_basis(const std::vector<occ::core::Atom> &atoms,
                          std::string name, bool cartesian = true) {
    occ::util::to_lower(name); // make name lowercase
    occ::io::JsonBasisReader reader(name);
    std::vector<occ::qm::OccShell> shells;
    for (const auto &atom : atoms) {
        const auto &element_basis = reader.element_basis(atom.atomic_number);
        for (const auto &electron_shell : element_basis.electron_shells) {
            for (size_t n = 0; n < electron_shell.angular_momentum.size();
                 n++) {
                shells.push_back(occ::qm::OccShell(
                    electron_shell.angular_momentum[n],
                    electron_shell.exponents, {electron_shell.coefficients[n]},
                    {atom.x, atom.y, atom.z}));
            }
        }
    }

    // normalize the shells
    for (auto &shell : shells) {
        if (!cartesian) {
            shell.kind = Kind::Spherical;
        }
        shell.incorporate_shell_norm();
    }
    return IntegralEngine(atoms, shells);
}

TEST_CASE("Water nuclear attraction", "[cint]") {
    using occ::qm::OccShell;
    libint2::Shell::do_enforce_unit_normalization(true);
    if (!libint2::initialized())
        libint2::initialize();
    occ::parallel::set_num_threads(1);

    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    occ::qm::BasisSet basis("def2-tzvp", atoms);

    constexpr Kind kind = Kind::Cartesian;
    basis.set_pure(kind == Kind::Spherical);

    occ::hf::HartreeFock hf(atoms, basis);
    occ::ints::FockBuilder fock(basis.max_nprim(), basis.max_l());

    IntegralEngine basis2(atoms, occ::qm::from_libint2_basis(basis));

    occ::Mat D = occ::Mat::Identity(basis2.nbf(), basis2.nbf());
    occ::qm::MolecularOrbitals mo;
    mo.D = D;
    occ::Mat f1 = fock.compute_fock<occ::qm::SpinorbitalKind::Restricted>(
        basis, hf.shellpair_list(), hf.shellpair_data(), mo);
    occ::Mat f2 = basis2.fock_operator<kind>(D);
    fmt::print("Fock max err: {}\n", (f2 - f1).cwiseAbs().maxCoeff());

    Mat schw1 = hf.compute_schwarz_ints();
    Mat schw2 = basis2.schwarz<kind>();
    fmt::print("Schwarz max err: {}\n", (schw2 - schw1).cwiseAbs().maxCoeff());

    occ::Mat3N pos(3, 6);
    pos << -4, -4, -4, 4, 4, 4, -4, -4, 4, 4, -4, -4, -4, 4, -4, 4, -4, 4;

    auto e1 = hf.electronic_electric_potential_contribution(
        occ::qm::SpinorbitalKind::Restricted, mo, pos);

    fmt::print("ESP:\n");
    std::cout << e1 << '\n';
    auto e2 = basis2.electric_potential<kind>(D, pos);
    fmt::print("ESP new:\n");
    std::cout << e2 << '\n';
    std::cout << e2.array() / e1.array() << '\n';
}
