#include "catch.hpp"
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>

TEST_CASE("Water DFT", "[scf]") {
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};
    auto basis = occ::qm::AOBasis::load(atoms, "def2-tzvp");
    basis.set_pure(false);
    auto hf = occ::hf::HartreeFock(basis);
    occ::scf::SCF<occ::hf::HartreeFock, occ::qm::SpinorbitalKind::Restricted>
        scf(hf);
    double e = scf.compute_scf_energy();

    occ::dft::AtomGridSettings settings;
    settings.max_angular_points = 302;
    settings.radial_precision = 1e-12;
    fmt::print("Construct\n");
    occ::dft::cosx::SemiNumericalExchange sgx(basis, settings);
    fmt::print("Construct done\n");

    occ::timing::StopWatch<2> sw;
    sw.start(0);
    fmt::print("Compute K SGX\n");
    occ::Mat result =
        sgx.compute_K(occ::qm::SpinorbitalKind::Restricted, scf.mo);
    sw.stop(0);
    fmt::print("Compute K SGX done\n");
    occ::Mat Jexact, Kexact;
    sw.start(1);
    std::tie(Jexact, Kexact) = hf.compute_JK(
        occ::qm::SpinorbitalKind::Restricted, scf.mo, 1e-12, occ::Mat());
    sw.stop(1);
    int i, j;
    fmt::print("K - Kexact: {:12.8f}\n",
               (result - Kexact).array().cwiseAbs().maxCoeff(&i, &j));
    /*
    const auto &bf2shell = sgx.engine().aobasis().bf_to_shell();
    int s1 = bf2shell[i];
    const auto &sh1 = sgx.engine().aobasis()[s1];
    int s2 = bf2shell[i];
    const auto &sh2 = sgx.engine().aobasis()[s2];

    fmt::print("Shells:\n");
    std::cout << s1 << sh1 << '\n' << s2 << sh2 << '\n';
    int bf1 = sgx.engine().aobasis().first_bf()[s1];
    int bf2 = sgx.engine().aobasis().first_bf()[s2];

    fmt::print("K SGX\n{}\n", result.block(bf1, bf2, sh1.size(), sh2.size()));
    fmt::print("K exact\n{}\n", Kexact.block(bf1, bf2, sh1.size(), sh2.size()));
    */
    fmt::print("Speedup = ({} vs. {}) {:.3f} times\n", sw.read(0), sw.read(1),
               sw.read(1) / sw.read(0));

    occ::timing::print_timings();
}
