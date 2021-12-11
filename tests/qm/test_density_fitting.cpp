#include "catch.hpp"
#include <occ/qm/density_fitting.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>

using occ::df::DFFockEngine;

TEST_CASE("H2O/6-31G") {
    libint2::Shell::do_enforce_unit_normalization(false);
    if (!libint2::initialized()) libint2::initialize();

    std::vector<occ::core::Atom> atoms{
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.39839733}
    };
    occ::qm::BasisSet basis("sto-3g", atoms);
    basis.set_pure(false);
    occ::qm::BasisSet dfbasis("def2-svp-jk", atoms);

    occ::Mat C(2, 2);
    C << 0.54884228,  1.21245192,
         0.54884228, -1.21245192;

    occ::Mat D = C.leftCols(1) * C.leftCols(1).transpose();
    fmt::print("D:\n{}\n", D);

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

    occ::Mat F = df.compute_2body_fock_dfC(C.leftCols(1));
    occ::Mat Japprox = df.compute_J(D);
    occ::Mat Japprox2 = df.compute_J_direct(D);
    fmt::print("F\n{}\nE = {}\n", F, occ::qm::expectation<occ::qm::SpinorbitalKind::Restricted>(D, Jexact));
    fmt::print("Jexact\n{}\n", Jexact);
    fmt::print("Japprox\n{}\n", Japprox);
    fmt::print("Japprox2\n{}\n", Japprox2);
    occ::Mat Kapprox;
    std::tie(Japprox, Kapprox) = df.compute_JK_direct(C.leftCols(1));
    fmt::print("Kapprox\n{}\n", Kapprox);
    fmt::print("Kexact\n{}\n", Kexact);
    fmt::print("Fexact\n{}\n", Fexact);
}
