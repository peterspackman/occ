#include "catch.hpp"
#include <occ/qm/hf.h>
#include <occ/qm/property_ints.h>
#include <occ/qm/basisset.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>

using occ::qm::BasisSet;
using occ::hf::HartreeFock;
using occ::util::all_close;

TEST_CASE("H2/STO-3G") {
    libint2::initialize();
    libint2::Shell::do_enforce_unit_normalization(false);
    std::vector<libint2::Atom> atoms {
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.398397}
    };
    BasisSet basis("sto-3g", atoms);
    occ::MatRM D(2, 2);
    D.setConstant(0.301228);
    auto grid_pts = occ::Mat3N(3, 4);
    grid_pts << 1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 1.0,
                0.0, 0.0, 1.0, 1.0;
    HartreeFock hf(atoms, basis);

    occ::Vec expected_esp = occ::Vec(4);
    expected_esp << -1.37628, -1.37628, -1.95486, -1.45387;

    auto field_values = hf.nuclear_electric_field_contribution(grid_pts);
    fmt::print("Grid points\n{}\n", grid_pts);
    fmt::print("Nuclear E field values:\n{}\n", field_values);

    occ::ints::shellpair_list_t shellpair_list;
    occ::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = occ::ints::compute_shellpairs(basis);

    auto esp = occ::ints::compute_electric_potential<occ::qm::SpinorbitalKind::Restricted>(D, basis, shellpair_list, grid_pts);
    fmt::print("ESP:\n{}\n", esp);
    REQUIRE(all_close(esp, expected_esp, 1e-5, 1e-5));
    occ::Mat expected_efield(field_values.rows(), field_values.cols());
    occ::Mat efield;

    expected_efield << -0.592642, 0.0, 0.0, 0.0,
                        0.0, -0.592642, 0.0, -0.652486,
                        0.26967, 0.26967, -0.0880444, -0.116878;

    double delta = 1e-8;
    occ::Mat3N efield_fd(field_values.rows(), field_values.cols());
    for(size_t i = 0; i < 3; i++) {
        auto grid_pts_d = grid_pts;
        grid_pts_d.row(i).array() += delta;
        auto esp_d = occ::ints::compute_electric_potential<occ::qm::SpinorbitalKind::Restricted>(D, basis, shellpair_list, grid_pts_d);
        efield_fd.row(i) = - (esp_d - esp) / delta;
    }
    REQUIRE(all_close(efield_fd, expected_efield, 1e-5, 1e-5));
    fmt::print("Electric field FD:\n{}\n", efield_fd);


    if constexpr(LIBINT2_MAX_DERIV_ORDER > 1) {
        efield = occ::ints::compute_electric_field<occ::qm::SpinorbitalKind::Restricted>(D, basis, shellpair_list, grid_pts);
        fmt::print("Electric field:\n{}\n", efield);
        REQUIRE(all_close(efield, expected_efield, 1e-5, 1e-5));
    }

}
