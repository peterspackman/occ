#include "catch.hpp"
#include <occ/qm/density_fitting.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/mo.h>

using occ::df::DFFockEngine;

TEST_CASE("H2O/6-31G") {
    libint2::Shell::do_enforce_unit_normalization(true);
    if (!libint2::initialized()) libint2::initialize();

    std::vector<occ::core::Atom> atoms{
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.39839733}
    };
    occ::qm::BasisSet basis("sto-3g", atoms);
    basis.set_pure(false);
    occ::qm::BasisSet dfbasis("def2-svp-jk", atoms);
    dfbasis.set_pure(true);

    occ::qm::MolecularOrbitals mo;

    mo.C = occ::Mat(2, 2);
    mo.C << 0.54884228,  1.21245192,
            0.54884228, -1.21245192;

    mo.Cocc = mo.C.leftCols(1);
    mo.D = mo.Cocc * mo.Cocc.transpose();
    fmt::print("D:\n{}\n", mo.D);

    occ::Mat Fexact(2, 2);
    Fexact << 1.50976125, 0.7301775,
              0.7301775 , 1.50976125;

    occ::Mat Jexact(2, 2);
    Jexact << 1.34575531, 0.89426314,
              0.89426314, 1.34575531;

    occ::Mat Kexact(2, 2);
    Kexact << 1.18164378, 1.05837468,
              1.05837468, 1.18164378;

    DFFockEngine df(basis, dfbasis);

    occ::Mat Japprox = df.compute_J(mo);
    occ::Mat Kapprox = df.compute_K(mo);
    occ::Mat F = 2 * (Japprox - Kapprox);
    fmt::print("Fexact\n{}\n", Fexact);
    fmt::print("Fapprox\n{}\n", F);

    fmt::print("Jexact\n{}\n", Jexact);
    fmt::print("Japprox\n{}\n", Japprox);

    std::tie(Japprox, Kapprox) = df.compute_JK(mo);
    fmt::print("Kexact\n{}\n", Kexact);
    fmt::print("Kapprox\n{}\n", 2 * Kapprox);
}
