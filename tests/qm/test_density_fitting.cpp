#include "catch.hpp"
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>

TEST_CASE("H2O/6-31G") {
    std::vector<occ::core::Atom> atoms{{1, 0.0, 0.0, 0.0},
                                       {1, 0.0, 0.0, 1.39839733}};
    occ::qm::BasisSet basis("sto-3g", atoms);
    basis.set_pure(false);
    auto hf = occ::hf::HartreeFock(atoms, basis);
    hf.set_density_fitting_basis("def2-svp-jk");

    occ::qm::MolecularOrbitals mo;
    auto sk = occ::qm::SpinorbitalKind::Restricted;

    mo.C = occ::Mat(2, 2);
    mo.C << 0.54884228, 1.21245192, 0.54884228, -1.21245192;

    mo.Cocc = mo.C.leftCols(1);
    mo.D = mo.Cocc * mo.Cocc.transpose();
    fmt::print("D:\n{}\n", mo.D);

    occ::Mat Fexact(2, 2);
    Fexact << 1.50976125, 0.7301775, 0.7301775, 1.50976125;

    occ::Mat Jexact(2, 2);
    Jexact << 1.34575531, 0.89426314, 0.89426314, 1.34575531;

    occ::Mat Kexact(2, 2);
    Kexact << 1.18164378, 1.05837468, 1.05837468, 1.18164378;

    occ::Mat Japprox, Kapprox;
    std::tie(Japprox, Kapprox) = hf.compute_JK(sk, mo);
    occ::Mat F = 2 * (Japprox - Kapprox);
    fmt::print("Fexact\n{}\n", Fexact);
    fmt::print("Fapprox\n{}\n", F);

    fmt::print("Jexact\n{}\n", Jexact);
    fmt::print("Japprox\n{}\n", Japprox);

    std::tie(Japprox, Kapprox) = hf.compute_JK(sk, mo);
    fmt::print("Kexact\n{}\n", Kexact);
    fmt::print("Kapprox\n{}\n", 2 * Kapprox);
}
