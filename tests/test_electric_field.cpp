#include "catch.hpp"
#include "hf.h"
#include "ints.h"
#include "basisset.h"
#include <fmt/ostream.h>
#include "util.h"

using tonto::qm::BasisSet;
using tonto::hf::HartreeFock;
using tonto::util::all_close;

TEST_CASE("H2/STO-3G") {
    libint2::initialize();
    libint2::Shell::do_enforce_unit_normalization(false);
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    BasisSet basis("sto-3g", atoms);
    tonto::MatRM D(2, 2);
    D.setConstant(0.301228);
    auto grid_pts = tonto::Mat3N(3, 4);
    grid_pts << 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 1.0;
    HartreeFock hf(atoms, basis);
    auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
    fmt::print("Grid points\n{}\n", grid_pts);
    fmt::print("Nuclear E field values:\n{}\n", field_values);

    tonto::ints::shellpair_list_t shellpair_list;
    tonto::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = tonto::ints::compute_shellpairs(basis);

    auto esp = tonto::ints::compute_electric_potential(D, basis, shellpair_list, grid_pts);
    fmt::print("ESP:\n{}\n", esp);
    auto efield = tonto::ints::compute_electric_field(D, basis, shellpair_list, grid_pts);
    fmt::print("Electric field:\n{}\n", efield);

    double delta = 1e-8;
    tonto::Mat3N efield_fd(efield.rows(), efield.cols());
    for(size_t i = 0; i < 3; i++) {
        auto grid_pts_d = grid_pts;
        grid_pts_d.row(i).array() += delta;
        auto esp_d = tonto::ints::compute_electric_potential(D, basis, shellpair_list, grid_pts_d);
        efield_fd.row(i) = (esp_d - esp) / delta;
    }
    fmt::print("Electric field FD:\n{}\n", efield_fd);
    REQUIRE(all_close(efield, efield_fd, 1e-5, 1e-5));
}
