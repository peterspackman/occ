#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "catch.hpp"
#include <occ/core/molecule.h>
#include <occ/qm/density_fitting.h>
#include <fmt/ostream.h>
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>
#include <occ/qm/scf.h>
#include <occ/qm/hf.h>
#include <occ/core/timings.h>

using occ::df::DFFockEngine;
using occ::hf::HartreeFock;
using occ::Mat;
using occ::core::Molecule;
using occ::qm::SpinorbitalKind::Restricted;
using occ::scf::SCF;

Molecule water() {
    occ::Mat3N pos(3, 3);
    occ::IVec nums(3);
    nums << 8, 1, 1;
    pos <<  -0.7021961, -0.0560603,  0.0099423,
            -1.0221932,  0.8467758, -0.0114887,
             0.2575211,  0.0421215,  0.0052190;
    return Molecule(nums, pos.transpose());
}

TEST_CASE("H2O/def2-svp") {
    libint2::Shell::do_enforce_unit_normalization(false);
    if (!libint2::initialized()) libint2::initialize();


    occ::timing::StopWatch sw;

    Molecule m = water();

    occ::qm::BasisSet basis("def2-qzvpp", m.atoms());
    basis.set_pure(false);
    // density fitting basis must be pure!
    occ::qm::BasisSet dfbasis("def2-svp-jk", m.atoms());
    dfbasis.set_pure(true);

    HartreeFock hf = HartreeFock(m.atoms(), basis);
    SCF<HartreeFock, Restricted> scf(hf);
    double e = scf.compute_scf_energy();
    DFFockEngine df(basis, dfbasis);


    Mat J_exact, K_exact;
    std::tie(J_exact, K_exact) = scf.m_procedure.compute_JK(Restricted, scf.mo);

    Mat J_approx, K_approx;
    auto tprev = sw.read();
    sw.start();
    J_approx = df.compute_J(scf.mo);
    K_approx = df.compute_K(scf.mo);
    Mat J_approx_direct = df.compute_J_direct(scf.mo);
    Mat K_approx_direct = df.compute_K_direct(scf.mo);
    sw.stop();
    fmt::print("nbf * nbf: {}, num_rows: {}\n", df.nbf * df.nbf, df.num_rows());
    fmt::print("Integral storage size: {:.3f} MiB\n", df.integral_storage_max_size() * sizeof(double) * 1.0 / (1024 * 1024));
    fmt::print("initial J matrix build took: {:.3f} ms\n", 1000 * (sw.read() - tprev));
    Eigen::Index i, j;
    double max_err = (J_approx - J_exact).array().abs().maxCoeff(&i, &j);
    fmt::print("Max error J({},{}) = {}\n", i, j, max_err);
    double max_err_k = (K_approx - K_exact).array().abs().maxCoeff(&i, &j); 
    fmt::print("Max error K({},{}) = {}\n", i, j, max_err_k);
    fmt::print("Ratio:\n{}\n", K_approx_direct.array() / K_approx.array());


    BENCHMARK("J exact") {
        J_exact = scf.m_procedure.compute_J(Restricted, scf.mo);
        return 0;
    };


    BENCHMARK("J density fitting") {
        J_approx = df.compute_J(scf.mo);
        return 0;
    };

    BENCHMARK("J density fitting direct") {
        J_approx = df.compute_J_direct(scf.mo);
        return 0;
    };

    BENCHMARK("JK exact") {
        std::tie(J_exact, K_exact) = scf.m_procedure.compute_JK(Restricted, scf.mo);
        return 0;
    };

    BENCHMARK("K density fitting") {
        K_approx = df.compute_K(scf.mo);
        return 0;
    };

    BENCHMARK("K density fitting direct") {
        K_approx = df.compute_K_direct(scf.mo);
        return 0;
    };
    
    BENCHMARK("JK density fitting") {
        std::tie(J_approx, K_approx) = df.compute_JK(scf.mo);
        return 0;
    };


}
