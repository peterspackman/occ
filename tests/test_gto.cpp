#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <vector>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "gto.h"
#include "density.h"

TEST_CASE("GTO vals H2/STO-3G") {
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    libint2::BasisSet basis("sto-3g", atoms);
    auto grid_pts = tonto::Mat::Identity(3, 4);
    auto gto_values = tonto::gto::evaluate_basis_on_grid<1>(basis, atoms, grid_pts);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);

    tonto::MatRM D(2, 2);
    D.setConstant(0.60245569);
    auto rho = tonto::density::evaluate_density_on_grid<1>(basis, atoms, D, grid_pts);
    fmt::print("Rho\n{}\n", rho);
}

TEST_CASE("GTO vals H2/3-21G") {
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    libint2::BasisSet basis("3-21G", atoms);
    auto grid_pts = tonto::Mat::Identity(3, 4);
    auto gto_values = tonto::gto::evaluate_basis_on_grid<1>(basis, atoms, grid_pts);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);

    tonto::MatRM D(4, 4);
    D << 0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
         0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128,
         0.175416203439, 0.181496024303, 0.175416203439, 0.181496024303,
         0.181496024303, 0.187786568128, 0.181496024303, 0.187786568128;

    auto rho = tonto::density::evaluate_density_on_grid<1>(basis, atoms, D, grid_pts);
    fmt::print("Rho\n{}\n", rho);
}

TEST_CASE("GTO vals H2/STO-3G Unrestricted") {
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    libint2::BasisSet basis("sto-3g", atoms);
    auto grid_pts = tonto::Mat::Identity(3, 4);
    auto gto_values = tonto::gto::evaluate_basis_on_grid<1>(basis, atoms, grid_pts);
    fmt::print("Gto values\nphi:\n{}\n", gto_values.phi);
    fmt::print("phi_x\n{}\n", gto_values.phi_x);
    fmt::print("phi_y\n{}\n", gto_values.phi_y);
    fmt::print("phi_z\n{}\n", gto_values.phi_z);

    tonto::MatRM D(4, 2);
    D.block(0, 0, 2, 2).setConstant(0.60245569);
    D.block(2, 0, 2, 2).setConstant(0.301277);
    auto rho = tonto::density::evaluate_density_on_grid<1, tonto::qm::SpinorbitalKind::Unrestricted>(basis, atoms, D, grid_pts);
    fmt::print("Rho alpha\n{}\n", rho.alpha());
    fmt::print("Rho beta\n{}\n", rho.beta());
}
