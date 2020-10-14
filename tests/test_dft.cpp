#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <libint2.hpp>
#include <vector>
#include <iostream>
#include "hf.h"
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "density.h"
#include "gto.h"
#include "util.h"

TEST_CASE("Water DFT grid", "[dft]")
{

    std::vector<libint2::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}
    };
    libint2::BasisSet basis("3-21G", atoms);

    SECTION("Grid generation") {
        tonto::dft::DFTGrid grid(basis, atoms);
        grid.set_min_angular_points(12);
        grid.set_max_angular_points(20);
        auto pts = grid.grid_points(0);
        assert(pts.rows() == 1564);
        auto hpts_a = grid.grid_points(1);
        auto hpts_b = grid.grid_points(1);
        assert(hpts_a.rows() == hpts_b.rows());
    }

    std::vector<libint2::Atom> atomsH2{
        {1, 0.0, 0.0, 0.0},
        {1, 0.0, 0.0, 1.7}
    };
    libint2::BasisSet basisH2("sto-3g", atomsH2);

    tonto::MatRM DH2 = tonto::MatRM::Constant(2, 2, 0.8394261);
    tonto::MatN4 pts(3, 4);
    pts << 1, 0, 0, 1,
            0, 1, 0, 1,
            0, 0, 1, 1;

    SECTION("Density H2/STO-3G") {
        tonto::Vec expected_rho(3);
        expected_rho << 0.07057915015926258, 0.07057915015926258, 0.2528812862521075;
        fmt::print("Evaluation points:\n{}\n", pts);
        auto rho = tonto::density::evaluate(basisH2, atomsH2, DH2, pts);
        fmt::print("desired: {:20.14f} {:20.14f} {:20.14f}\n", expected_rho(0), expected_rho(1), expected_rho(2));
        fmt::print("found:   {:20.14f} {:20.14f} {:20.14f}\n", rho(0), rho(1), rho(2));
        REQUIRE(tonto::util::all_close(expected_rho, rho));
    }

    SECTION("Density H2O/3-21G") {
        tonto::Mat D_h2o(13, 13);
        fmt::print("Evaluation points:\n{}\n", pts);
        D_h2o <<
                 1.02716, 0.0370403, 0.0126073, 0.01989, -0.000519431, -0.235416, 0.00892977, 0.0141049, -0.000368326, -0.0124862, 0.0119227, -0.0122251, 0.0116199,
                0.0370403, 0.0633479, -0.00975063, -0.0151089, 0.00039499, 0.183785, -0.0118527, -0.0186074, 0.000486075, 0.0148647, -0.00396093, 0.0144757, -0.00394649,
                0.0126073, -0.00975063, 0.170367, 0.0190324, -0.000223437, -0.0639489, 0.17141, 0.0374756, -0.000541062, -0.0422867, -0.0362331, 0.114164, 0.0889713,
                0.01989, -0.0151089, 0.0190324, 0.189488, 0.00198374, -0.102031, 0.0373963, 0.207188, 0.0029189, 0.107343, 0.0801187, 0.00764228, 0.00177756,
                -0.000519431, 0.00039499, -0.000223437, 0.00198374, 0.271974, 0.00266281, -0.000539113, 0.00291871, 0.329479, -0.00253785, -0.00188296, -0.000462594, -0.000257767,
                -0.235416, 0.183785, -0.0639489, -0.102031, 0.00266281, 0.666584, -0.0737847, -0.117824, 0.00307484, 0.0286, -0.0303368, 0.0285497, -0.0295861,
                0.00892977, -0.0118527, 0.17141, 0.0373963, -0.000539113, -0.0737847, 0.174271, 0.0575285, -0.000873375, -0.0316797, -0.0282868, 0.11431, 0.0886869,
                0.0141049, -0.0186074, 0.0374756, 0.207188, 0.00291871, -0.117824, 0.0575285, 0.228282, 0.00406553, 0.111944, 0.0831599, 0.0194102, 0.0106733,
                -0.000368326, 0.000486075, -0.000541062, 0.0029189, 0.329479, 0.00307484, -0.000873375, 0.00406553, 0.399144, -0.00267651, -0.00197686, -0.000752321, -0.000475831,
                -0.0124862, 0.0148647, -0.0422867, 0.107343, -0.00253785, 0.0286, -0.0316797, 0.111944, -0.00267651, 0.0857444, 0.0592602, -0.0233096, -0.0269691,
                0.0119227, -0.00396093, -0.0362331, 0.0801187, -0.00188296, -0.0303368, -0.0282868, 0.0831599, -0.00197686, 0.0592602, 0.0455832, -0.0266874, -0.0225519,
                -0.0122251, 0.0144757, 0.114164, 0.00764228, -0.000462594, 0.0285497, 0.11431, 0.0194102, -0.000752321, -0.0233096, -0.0266874, 0.0846396, 0.0598386,
                0.0116199, -0.00394649, 0.0889713, 0.00177756, -0.000257767, -0.0295861, 0.0886869, 0.0106733, -0.000475831, -0.0269691, -0.0225519, 0.0598386, 0.0468541;
        tonto::Vec expected_rho(3);
        expected_rho << 0.06179271596276, 0.04806065376604, 0.05292253394790;
        auto rho = tonto::density::evaluate(basis, atoms, D_h2o, pts);
        fmt::print("desired: {:20.14f} {:20.14f} {:20.14f}\n", expected_rho(0), expected_rho(1), expected_rho(2));
        fmt::print("found:   {:20.14f} {:20.14f} {:20.14f}\n", rho(0), rho(1), rho(2));
        REQUIRE(tonto::util::all_close(expected_rho, rho));
    }

    SECTION("Density integral H2O/3-21G") {
        tonto::Mat D_h2o(13, 13);
        D_h2o <<
                 1.02716, 0.0370403, 0.0126073, 0.01989, -0.000519431, -0.235416, 0.00892977, 0.0141049, -0.000368326, -0.0124862, 0.0119227, -0.0122251, 0.0116199,
                0.0370403, 0.0633479, -0.00975063, -0.0151089, 0.00039499, 0.183785, -0.0118527, -0.0186074, 0.000486075, 0.0148647, -0.00396093, 0.0144757, -0.00394649,
                0.0126073, -0.00975063, 0.170367, 0.0190324, -0.000223437, -0.0639489, 0.17141, 0.0374756, -0.000541062, -0.0422867, -0.0362331, 0.114164, 0.0889713,
                0.01989, -0.0151089, 0.0190324, 0.189488, 0.00198374, -0.102031, 0.0373963, 0.207188, 0.0029189, 0.107343, 0.0801187, 0.00764228, 0.00177756,
                -0.000519431, 0.00039499, -0.000223437, 0.00198374, 0.271974, 0.00266281, -0.000539113, 0.00291871, 0.329479, -0.00253785, -0.00188296, -0.000462594, -0.000257767,
                -0.235416, 0.183785, -0.0639489, -0.102031, 0.00266281, 0.666584, -0.0737847, -0.117824, 0.00307484, 0.0286, -0.0303368, 0.0285497, -0.0295861,
                0.00892977, -0.0118527, 0.17141, 0.0373963, -0.000539113, -0.0737847, 0.174271, 0.0575285, -0.000873375, -0.0316797, -0.0282868, 0.11431, 0.0886869,
                0.0141049, -0.0186074, 0.0374756, 0.207188, 0.00291871, -0.117824, 0.0575285, 0.228282, 0.00406553, 0.111944, 0.0831599, 0.0194102, 0.0106733,
                -0.000368326, 0.000486075, -0.000541062, 0.0029189, 0.329479, 0.00307484, -0.000873375, 0.00406553, 0.399144, -0.00267651, -0.00197686, -0.000752321, -0.000475831,
                -0.0124862, 0.0148647, -0.0422867, 0.107343, -0.00253785, 0.0286, -0.0316797, 0.111944, -0.00267651, 0.0857444, 0.0592602, -0.0233096, -0.0269691,
                0.0119227, -0.00396093, -0.0362331, 0.0801187, -0.00188296, -0.0303368, -0.0282868, 0.0831599, -0.00197686, 0.0592602, 0.0455832, -0.0266874, -0.0225519,
                -0.0122251, 0.0144757, 0.114164, 0.00764228, -0.000462594, 0.0285497, 0.11431, 0.0194102, -0.000752321, -0.0233096, -0.0266874, 0.0846396, 0.0598386,
                0.0116199, -0.00394649, 0.0889713, 0.00177756, -0.000257767, -0.0295861, 0.0886869, 0.0106733, -0.000475831, -0.0269691, -0.0225519, 0.0598386, 0.0468541;
        tonto::dft::DFTGrid grid(basis, atoms);
        double total_density = 0.0;

        for(size_t i = 0; i < atoms.size(); i++) {
            auto pts = grid.grid_points(i);
            auto rho = tonto::density::evaluate(basis, atoms, D_h2o, pts);
            total_density += rho.col(0).dot(pts.col(3));
        }
        fmt::print("Integrated n_e: {:20.14f} (should be 10)\n", 2 * total_density);
        REQUIRE(total_density * 2 == Approx(10.0));
    }

    SECTION("DFT LDA") {
        tonto::dft::DensityFunctional::Params params(3);
        params.rho << 1.0, 0.5, 0.3;
        tonto::Array expected(3);
        expected << -0.7385587663820224, -0.586194481347579, -0.49441557378816503;
        fmt::print("LDA exchange functional\n");
        tonto::dft::DensityFunctional lda("xc_lda_x");
        REQUIRE(lda.family_string() == "LDA");
        REQUIRE(lda.kind_string() == "exchange");
        auto e = lda.evaluate(params);
        fmt::print("desired: {:20.14f} {:20.14f} {:20.14f}\n", expected(0), expected(1), expected(2));
        fmt::print("found:   {:20.14f} {:20.14f} {:20.14f}\n", e.exc(0), e.exc(1), e.exc(2));
        REQUIRE(tonto::util::all_close(e.exc, expected));
        tonto::dft::DensityFunctional::Params params2;
        params2.rho = tonto::Array::Random(10, 1);
        params2.rho = params2.rho.abs();
        auto v = lda.evaluate(params2);
        fmt::print("rho2:\n{}\n", params2.rho);
        fmt::print("potential:\n{}\n", v.vrho);
    }

    SECTION("GTO vals H2/STO-3G") {
        std::vector<libint2::Atom> atoms {
            {1, 0.0, 0.0, 0.0},
            {1, 0.0, 0.0, 1.398397}
        };
        libint2::BasisSet basis("sto-3g", atoms);
        tonto::Mat D(2, 2);
        D.setConstant(0.6024556858490756);
        fmt::print("Evaluation points\n{}\n", pts);
        auto gto_vals = tonto::density::evaluate_gtos(basis, atoms, pts, 1);
        tonto::Mat expected(2, 12);
        expected << 0.22303583, -0.26463287, 0, 0, 0.22303583, 0, -0.26463287, 0, 0.22303583, 0, 0, -0.26463287,
                    0.09305658, -0.07005863, 0, 0.9796981, 0.09305658, 0, -0.07005863, 0.9796981, 0.48464641, 0, 0, 0.57037312;
        fmt::print("Expected:\n{}\n", expected.transpose());
        fmt::print("GTO values H2/STO-3G:\n{}\n", gto_vals.transpose());
        auto phi = Eigen::Map<tonto::Mat, 0, Eigen::OuterStride<>>(gto_vals.data(), gto_vals.rows(), pts.rows(), {4 * gto_vals.rows()});
        auto phi_x = Eigen::Map<tonto::Mat, 0, Eigen::OuterStride<>>(gto_vals.data() + gto_vals.rows(), gto_vals.rows(), pts.rows(), {4 * gto_vals.rows()});
        auto phi_y = Eigen::Map<tonto::Mat, 0, Eigen::OuterStride<>>(gto_vals.data() + 2*gto_vals.rows(), gto_vals.rows(), pts.rows(), {4 * gto_vals.rows()});
        auto phi_z = Eigen::Map<tonto::Mat, 0, Eigen::OuterStride<>>(gto_vals.data() + 3*gto_vals.rows(), gto_vals.rows(), pts.rows(), {4 * gto_vals.rows()});

        fmt::print("phi:\n{}\n", phi.transpose());
        fmt::print("phi_x:\n{}\n", phi_x.transpose());
        fmt::print("phi_y:\n{}\n", phi_y.transpose());
        fmt::print("phi_z:\n{}\n", phi_z.transpose());

    }

    SECTION("Density Gradients H2/3-21G") {
        std::vector<libint2::Atom> atoms{
            {1, 0.0, 0.0, 0.0},
            {1, 0.0, 0.0, 1.398397}
        };
        libint2::BasisSet basis("3-21g", atoms);
        tonto::Mat D(4, 4), expected(4, 3);
        D << 0.1736564,  0.18149133, 0.1736564, 0.18149133,
             0.18149133, 0.187831,   0.18149133, 0.187831,
             0.1736564,  0.18149133, 0.1736564, 0.18149133,
             0.18149133, 0.187831,   0.18149133, 0.187831;
        fmt::print("Evaluation points:\n{}\n", pts);
        expected << 0.06047766, 0.06047766, 0.28538242,
                    -0.12553876, 0, 0,
                    0, -0.12553876, 0,
                    0.03585136, 0.03585136, 0.26484129;
        auto rho = tonto::density::evaluate(basis, atoms, D, pts, 1);
        fmt::print("Expected:\n{}\n", expected);
        fmt::print("found:\n{}\n", rho.transpose());
        tonto::dft::DensityFunctional func("xc_gga_x_pbe");
        tonto::dft::DensityFunctional::Params params;
        params.rho = rho.col(0);
        const auto& rho_x = rho.col(1).array(), rho_y = rho.col(2).array(), rho_z = rho.col(3).array();
        params.sigma = rho_x * rho_x + rho_y * rho_y + rho_z * rho_z;
        auto res = func.evaluate(params);
        fmt::print("exc:\n{}\n", res.exc);
        fmt::print("vrho:\n{}\n", res.vrho);
        fmt::print("vsigma:\n{}\n", res.vsigma);
    }

    SECTION("Functionals") {
        tonto::dft::DensityFunctional func("xc_gga_x_pbe");
        REQUIRE(func.id() == tonto::dft::DensityFunctional::Identifier::gga_x_pbe);
    }
/*
    SECTION("Benchmarks"){
        BENCHMARK_ADVANCED("Density grid construction")(Catch::Benchmark::Chronometer meter) {
            tonto::Mat D_h2o(13, 13);
            meter.measure([&basis, &atoms] {
                tonto::dft::DFTGrid grid(basis, atoms);
                int num_pts{0};

                for(size_t i = 0; i < atoms.size(); i++) {
                    auto pts = grid.grid_points(i);
                    num_pts += pts.rows();
                }
            });
        };

        BENCHMARK_ADVANCED("Density evaluation")(Catch::Benchmark::Chronometer meter) {
            tonto::Mat D_h2o(13, 13);
            D_h2o <<
                     1.02716, 0.0370403, 0.0126073, 0.01989, -0.000519431, -0.235416, 0.00892977, 0.0141049, -0.000368326, -0.0124862, 0.0119227, -0.0122251, 0.0116199,
                    0.0370403, 0.0633479, -0.00975063, -0.0151089, 0.00039499, 0.183785, -0.0118527, -0.0186074, 0.000486075, 0.0148647, -0.00396093, 0.0144757, -0.00394649,
                    0.0126073, -0.00975063, 0.170367, 0.0190324, -0.000223437, -0.0639489, 0.17141, 0.0374756, -0.000541062, -0.0422867, -0.0362331, 0.114164, 0.0889713,
                    0.01989, -0.0151089, 0.0190324, 0.189488, 0.00198374, -0.102031, 0.0373963, 0.207188, 0.0029189, 0.107343, 0.0801187, 0.00764228, 0.00177756,
                    -0.000519431, 0.00039499, -0.000223437, 0.00198374, 0.271974, 0.00266281, -0.000539113, 0.00291871, 0.329479, -0.00253785, -0.00188296, -0.000462594, -0.000257767,
                    -0.235416, 0.183785, -0.0639489, -0.102031, 0.00266281, 0.666584, -0.0737847, -0.117824, 0.00307484, 0.0286, -0.0303368, 0.0285497, -0.0295861,
                    0.00892977, -0.0118527, 0.17141, 0.0373963, -0.000539113, -0.0737847, 0.174271, 0.0575285, -0.000873375, -0.0316797, -0.0282868, 0.11431, 0.0886869,
                    0.0141049, -0.0186074, 0.0374756, 0.207188, 0.00291871, -0.117824, 0.0575285, 0.228282, 0.00406553, 0.111944, 0.0831599, 0.0194102, 0.0106733,
                    -0.000368326, 0.000486075, -0.000541062, 0.0029189, 0.329479, 0.00307484, -0.000873375, 0.00406553, 0.399144, -0.00267651, -0.00197686, -0.000752321, -0.000475831,
                    -0.0124862, 0.0148647, -0.0422867, 0.107343, -0.00253785, 0.0286, -0.0316797, 0.111944, -0.00267651, 0.0857444, 0.0592602, -0.0233096, -0.0269691,
                    0.0119227, -0.00396093, -0.0362331, 0.0801187, -0.00188296, -0.0303368, -0.0282868, 0.0831599, -0.00197686, 0.0592602, 0.0455832, -0.0266874, -0.0225519,
                    -0.0122251, 0.0144757, 0.114164, 0.00764228, -0.000462594, 0.0285497, 0.11431, 0.0194102, -0.000752321, -0.0233096, -0.0266874, 0.0846396, 0.0598386,
                    0.0116199, -0.00394649, 0.0889713, 0.00177756, -0.000257767, -0.0295861, 0.0886869, 0.0106733, -0.000475831, -0.0269691, -0.0225519, 0.0598386, 0.0468541;
            meter.measure([&D_h2o, &basis, &atoms] {
                tonto::dft::DFTGrid grid(basis, atoms);
                double total_density = 0.0;

                for(size_t i = 0; i < atoms.size(); i++) {
                    auto pts = grid.grid_points(i);
                    auto rho = tonto::density::evaluate(basis, atoms, D_h2o, pts);
                    total_density += rho.dot(pts.col(3));
                }
            });
        };
    }
*/
}
