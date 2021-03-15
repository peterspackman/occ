#include "catch.hpp"
#include <tonto/qm/hf.h>
#include <tonto/qm/ints.h>
#include <tonto/qm/basisset.h>
#include <fmt/ostream.h>
#include <tonto/core/util.h>

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

    tonto::Vec expected_esp = tonto::Vec(4);
    expected_esp << -1.37628, -1.37628, -1.95486, -1.45387;

    auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
    fmt::print("Grid points\n{}\n", grid_pts);
    fmt::print("Nuclear E field values:\n{}\n", field_values);

    tonto::ints::shellpair_list_t shellpair_list;
    tonto::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = tonto::ints::compute_shellpairs(basis);

    auto esp = tonto::ints::compute_electric_potential(D, basis, shellpair_list, grid_pts);
    fmt::print("ESP:\n{}\n", esp);
    REQUIRE(all_close(esp, expected_esp, 1e-5, 1e-5));

    tonto::Mat expected_efield(field_values.rows(), field_values.cols());
    tonto::Mat efield;

    expected_efield << -0.592642, 0.0, 0.0, 0.0,
                        0.0, -0.592642, 0.0, -0.652486,
                        0.26967, 0.26967, -0.0880444, -0.116878;

    double delta = 1e-8;
    tonto::Mat3N efield_fd(field_values.rows(), field_values.cols());
    for(size_t i = 0; i < 3; i++) {
        auto grid_pts_d = grid_pts;
        grid_pts_d.row(i).array() += delta;
        auto esp_d = tonto::ints::compute_electric_potential(D, basis, shellpair_list, grid_pts_d);
        efield_fd.row(i) = - (esp_d - esp) / delta;
    }
    REQUIRE(all_close(efield_fd, expected_efield, 1e-5, 1e-5));
    fmt::print("Electric field FD:\n{}\n", efield_fd);


    if constexpr(LIBINT2_MAX_DERIV_ORDER > 1) {
        efield = tonto::ints::compute_electric_field(D, basis, shellpair_list, grid_pts);
        fmt::print("Electric field:\n{}\n", efield);
        REQUIRE(all_close(efield, expected_efield, 1e-5, 1e-5));
    }

}
