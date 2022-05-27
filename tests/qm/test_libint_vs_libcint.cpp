#include "catch.hpp"
#include <fmt/ostream.h>
#include <iostream>
#include <occ/core/parallel.h>
#include <occ/io/json_basis.h>
#include <occ/qm/basisset.h>
#include <occ/qm/fock.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <vector>

using occ::qm::cint::Operator;
using Kind = occ::qm::OccShell::Kind;
using occ::Mat;
using occ::qm::IntegralEngine;
using occ::qm::IntegralEngineDF;

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
    constexpr Kind kind = Kind::Spherical;
    basis.set_pure(kind == Kind::Spherical);
    auto basis2 = occ::qm::from_libint2_basis(basis);
    occ::qm::BasisSet dfbasis("def2-svp-jk", atoms);
    dfbasis.set_pure(kind == Kind::Spherical);
    auto dfbasis2 = occ::qm::from_libint2_basis(dfbasis);

    occ::hf::HartreeFock hf(atoms, basis);
    occ::ints::FockBuilder fock(basis.max_nprim(), basis.max_l());

    IntegralEngine engine(atoms, basis2);

    occ::Mat D = occ::Mat::Identity(engine.nbf(), engine.nbf());
    occ::qm::MolecularOrbitals mo;
    mo.D = D;
    occ::Mat f1 = fock.compute_fock<occ::qm::SpinorbitalKind::Restricted>(
        basis, hf.shellpair_list(), hf.shellpair_data(), mo);
    occ::Mat f2 = engine.fock_operator<kind>(D);
    fmt::print("Fock max err: {}\n", (f2 - f1).cwiseAbs().maxCoeff());

    Mat schw1 = hf.compute_schwarz_ints();
    Mat schw2 = engine.schwarz<kind>();
    fmt::print("Schwarz max err: {}\n", (schw2 - schw1).cwiseAbs().maxCoeff());

    occ::Mat3N pos(3, 6);
    pos << -4, -4, -4, 4, 4, 4, -4, -4, 4, 4, -4, -4, -4, 4, -4, 4, -4, 4;
    std::vector<occ::core::PointCharge> chgs;
    for (size_t i = 0; i < 6; i++) {
        chgs.push_back({i, {pos(0, i), pos(1, i), pos(2, i)}});
    }

    auto pc1 = hf.compute_point_charge_interaction_matrix(chgs);
    auto pc2 = engine.point_charge_potential<kind>(chgs);
    fmt::print("Point charge max err: {}\n", (pc2 - pc1).cwiseAbs().maxCoeff());
    std::cout << pc1.block(0, 0, 5, 5) << '\n' << pc2.block(0, 0, 5, 5) << '\n';

    auto e1 = hf.electronic_electric_potential_contribution(
        occ::qm::SpinorbitalKind::Restricted, mo, pos);
    auto e2 = engine.electric_potential<kind>(D, pos);
    fmt::print("ESP max err: {}\n", (e2 - e1).cwiseAbs().maxCoeff());

    hf.set_density_fitting_basis("def2-svp-jk");
    auto j1 = hf.compute_J(occ::qm::SpinorbitalKind::Restricted, mo);
    engine.set_auxiliary_basis(dfbasis2);
    // auto j2 = engine.coulomb_operator_df<kind>(D);
    auto V1 = occ::ints::compute_2body_2index_ints(dfbasis);
    IntegralEngine engine_aux(atoms, dfbasis2);
    auto V2 =
        engine_aux
            .one_electron_operator<occ::qm::cint::Operator::coulomb, kind>();
    fmt::print("2c2e max err: {}\n", (V2 - V1).cwiseAbs().maxCoeff());

    IntegralEngineDF engine_df(atoms, basis2, dfbasis2);
    auto J2 = engine_df.compute_coulomb_operator<kind>(D);
    fmt::print("J\n");
    std::cout << J2.block(0, 0, 10, 10) << '\n';
}
