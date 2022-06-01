#include "catch.hpp"
#include <occ/core/util.h>
#include <occ/dft/dft.h>
#include <occ/dft/seminumerical_exchange.h>
#include <occ/qm/scf.h>

TEST_CASE("Water DFT", "[scf]") {
    libint2::Shell::do_enforce_unit_normalization(true);
    if (!libint2::initialized())
        libint2::initialize();
    std::vector<occ::core::Atom> atoms{
        {8, -1.32695761, -0.10593856, 0.01878821},
        {1, -1.93166418, 1.60017351, -0.02171049},
        {1, 0.48664409, 0.07959806, 0.00986248}};

    occ::qm::BasisSet obs("def2-qzvpp", atoms);
    obs.set_pure(false);
    auto dft =
        occ::dft::DFT("lda", obs, atoms, occ::qm::SpinorbitalKind::Restricted);
    occ::scf::SCF<occ::dft::DFT, occ::qm::SpinorbitalKind::Restricted> scf(dft);
    double e = scf.compute_scf_energy();

    occ::dft::cosx::SemiNumericalExchange sgx(atoms, obs);

    occ::timing::StopWatch<2> sw;
    sw.start(0);
    occ::Mat result =
        sgx.compute_K(occ::qm::SpinorbitalKind::Restricted, scf.mo);
    sw.stop(0);
    occ::Mat Jexact, Kexact;
    sw.start(1);
    std::tie(Jexact, Kexact) = dft.hf().compute_JK(
        occ::qm::SpinorbitalKind::Restricted, scf.mo, 1e-12, occ::Mat());
    sw.stop(1);
    fmt::print("K - Kexact: {:12.8f}\n",
               (result - Kexact).array().cwiseAbs().maxCoeff());
    fmt::print("Speedup = ({} vs. {}) {:.3f} times\n", sw.read(0), sw.read(1),
               sw.read(1) / sw.read(0));

    occ::timing::print_timings();
}
