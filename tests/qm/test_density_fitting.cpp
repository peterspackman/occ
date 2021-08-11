#include "catch.hpp"
#include <occ/qm/density_fitting.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>

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

    occ::Mat Fexact(2, 2);
    Fexact << 1.50976125, 0.7301775,
              0.7301775 , 1.50976125;

    DFFockEngine df(basis, dfbasis);

    occ::Mat F = df.compute_2body_fock_dfC(C.leftCols(1));
    fmt::print("F\n{}\n", 2 * F);
    fmt::print("Fexact\n{}\n", Fexact);
}
