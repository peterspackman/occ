#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/parallel.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo.h>
#include <occ/qm/split_ri_j.h>
#include <thread>

// Benchmark comparing Traditional RI-J (stored/direct) vs Split-RI-J
TEST_CASE("Benchmark: Coulomb methods H2O/def2-SVP", "[.benchmark][bench-split-ri-j]") {
    // Set thread count to hardware concurrency
    int nthreads = std::thread::hardware_concurrency();
    occ::parallel::set_num_threads(nthreads);
    std::vector<occ::core::Atom> atoms{
        {8, 0.0000000, 0.0000000, 0.1173470},
        {1, 0.0000000, 0.7572150, -0.4693880},
        {1, 0.0000000, -0.7572150, -0.4693880}
    };

    auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
    basis.set_pure(false);
    auto aux_basis = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
    aux_basis.set_kind(basis.kind());

    const size_t nbf = basis.nbf();
    const size_t naux = aux_basis.nbf();

    occ::qm::MolecularOrbitals mo;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;
    mo.D = occ::Mat::Identity(nbf, nbf) * 0.1;

    occ::qm::IntegralEngine engine(basis);
    occ::qm::IntegralEngineDF df_engine(atoms, basis.shells(), aux_basis.shells());
    occ::qm::SplitRIJ split_rij(basis, aux_basis, engine.shellpairs(), engine.schwarz());

    fmt::print("\nBenchmark: H2O/def2-SVP ({} threads)\n", nthreads);
    fmt::print("  AO basis: {} functions, Aux basis: {} functions\n", nbf, naux);

    // Verify correctness first
    df_engine.set_coulomb_method(occ::qm::CoulombMethod::Traditional);
    df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Stored);
    auto J_stored = df_engine.coulomb(mo);

    df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Direct);
    auto J_direct = df_engine.coulomb(mo);

    auto J_split = split_rij.coulomb(mo);

    double diff_stored_direct = (J_stored - J_direct).cwiseAbs().maxCoeff();
    double diff_stored_split = (J_stored - J_split).cwiseAbs().maxCoeff();

    fmt::print("  Max diff (stored vs direct): {:12.2e}\n", diff_stored_direct);
    fmt::print("  Max diff (stored vs split):  {:12.2e}\n", diff_stored_split);

    REQUIRE(diff_stored_direct < 1e-10);
    REQUIRE(diff_stored_split < 1e-6);

    BENCHMARK("Traditional RI-J (stored)") {
        df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Stored);
        return df_engine.coulomb(mo);
    };

    BENCHMARK("Traditional RI-J (direct)") {
        df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Direct);
        return df_engine.coulomb(mo);
    };

    BENCHMARK("Split-RI-J") {
        return split_rij.coulomb(mo);
    };
}

TEST_CASE("Benchmark: Coulomb methods Benzene/def2-SVP", "[.benchmark][bench-split-ri-j][slow]") {
    // Set thread count to hardware concurrency
    int nthreads = std::thread::hardware_concurrency();
    occ::parallel::set_num_threads(nthreads);

    std::vector<occ::core::Atom> atoms{
        {6,  0.0000000,  1.3970000,  0.0000000},
        {6,  1.2098079,  0.6985000,  0.0000000},
        {6,  1.2098079, -0.6985000,  0.0000000},
        {6,  0.0000000, -1.3970000,  0.0000000},
        {6, -1.2098079, -0.6985000,  0.0000000},
        {6, -1.2098079,  0.6985000,  0.0000000},
        {1,  0.0000000,  2.4810000,  0.0000000},
        {1,  2.1486254,  1.2405000,  0.0000000},
        {1,  2.1486254, -1.2405000,  0.0000000},
        {1,  0.0000000, -2.4810000,  0.0000000},
        {1, -2.1486254, -1.2405000,  0.0000000},
        {1, -2.1486254,  1.2405000,  0.0000000}
    };

    auto basis = occ::gto::AOBasis::load(atoms, "def2-svp");
    basis.set_pure(false);
    auto aux_basis = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
    aux_basis.set_kind(basis.kind());

    const size_t nbf = basis.nbf();
    const size_t naux = aux_basis.nbf();

    occ::qm::MolecularOrbitals mo;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;
    mo.D = occ::Mat::Identity(nbf, nbf) * 0.1;

    occ::qm::IntegralEngine engine(basis);
    occ::qm::IntegralEngineDF df_engine(atoms, basis.shells(), aux_basis.shells());
    occ::qm::SplitRIJ split_rij(basis, aux_basis, engine.shellpairs(), engine.schwarz());

    fmt::print("\nBenchmark: Benzene/def2-SVP ({} threads)\n", nthreads);
    fmt::print("  AO basis: {} functions, Aux basis: {} functions\n", nbf, naux);

    // Verify correctness first
    df_engine.set_coulomb_method(occ::qm::CoulombMethod::Traditional);
    df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Stored);
    auto J_stored = df_engine.coulomb(mo);

    df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Direct);
    auto J_direct = df_engine.coulomb(mo);

    auto J_split = split_rij.coulomb(mo);

    double diff_stored_direct = (J_stored - J_direct).cwiseAbs().maxCoeff();
    double diff_stored_split = (J_stored - J_split).cwiseAbs().maxCoeff();

    fmt::print("  Max diff (stored vs direct): {:12.2e}\n", diff_stored_direct);
    fmt::print("  Max diff (stored vs split):  {:12.2e}\n", diff_stored_split);

    REQUIRE(diff_stored_direct < 1e-10);
    REQUIRE(diff_stored_split < 1e-6);

    BENCHMARK("Traditional RI-J (direct)") {
        df_engine.set_integral_policy(occ::qm::IntegralEngineDF::Policy::Direct);
        return df_engine.coulomb(mo);
    };

    BENCHMARK("Split-RI-J") {
        return split_rij.coulomb(mo);
    };
}
