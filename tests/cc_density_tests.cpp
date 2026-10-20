#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/units.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/density.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <array>
#include <vector>

using Catch::Matchers::WithinAbs;
using occ::qm::cc::CorrelatedDensityOptions;

// Water cc-pVDZ (spherical), all electrons. References: PySCF RHF 1e-12, CCSD
// 1e-11, ccsd_lambda 1e-10 and make_rdm1; CCSD(T) from ccsd_t,
// ccsd_t_lambda_slow and ccsd_t_rdm_slow (for_grad=True); relaxed MP2 from
// ORCA 6.1.0 (! MP2 NoRI, %mp2 density relaxed, Z_Tol 1e-9); MP3/MP4 from
// ORCA 6.1.0 AutoCI (! AUTOCI-MP4 NoRI).

namespace {

occ::qm::Wavefunction water_rhf() {
  constexpr double A = occ::units::ANGSTROM_TO_BOHR;
  const std::vector<occ::core::Atom> atoms{{8, 0.0, 0.0, 0.1173 * A},
                                           {1, 0.0, 0.7572 * A, -0.4692 * A},
                                           {1, 0.0, -0.7572 * A, -0.4692 * A}};
  auto basis = occ::gto::AOBasis::load(atoms, "cc-pvdz");
  basis.set_pure(true);
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  scf.convergence_settings.energy_threshold = 1e-12;
  scf.convergence_settings.commutator_threshold = 1e-9;
  scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  return scf.wavefunction();
}

const occ::qm::Wavefunction &rhf() {
  static const occ::qm::Wavefunction wfn = water_rhf();
  return wfn;
}

CorrelatedDensityOptions tight_all_electron() {
  CorrelatedDensityOptions opts;
  opts.n_frozen = 0;
  opts.tol = 1e-9;
  return opts;
}

constexpr std::array<double, 24> ccsd_occ{
    1.999916791935, 1.984591127484, 1.973055495266, 1.966977132722,
    1.964441817473, 0.026576000090, 0.024606534618, 0.016958989231,
    0.011482604142, 0.005877291267, 0.005610198708, 0.005102431406,
    0.004501850550, 0.004391650529, 0.001153986147, 0.001135571676,
    0.000733257552, 0.000666305288, 0.000595994224, 0.000541860862,
    0.000498173499, 0.000488628631, 0.000056697599, 0.000039609101};

constexpr std::array<double, 24> ccsd_t_occ{
    1.999915560472, 1.984126920375, 1.971987096386, 1.965109636192,
    1.962228897254, 0.028145280442, 0.025910267256, 0.017475185700,
    0.011824173422, 0.006209180127, 0.005911724793, 0.005345731873,
    0.004639591047, 0.004522823917, 0.001290017319, 0.001240118379,
    0.000812346188, 0.000788191094, 0.000716511774, 0.000597192399,
    0.000545450072, 0.000541000350, 0.000069385498, 0.000047717671};

constexpr std::array<double, 24> mp2_occ{
    1.99990672, 1.98705192, 1.97412587, 1.97059053, 1.96862196, 0.02309666,
    0.02088708, 0.01730897, 0.01033580, 0.00549425, 0.00515392, 0.00470589,
    0.00410811, 0.00400362, 0.00092385, 0.00086708, 0.00061265, 0.00051036,
    0.00045376, 0.00045212, 0.00037840, 0.00034817, 0.00003161, 0.00003068};

template <size_t N>
void check_occupations(const occ::Vec &n, const std::array<double, N> &ref,
                       const double tol) {
  REQUIRE(n.size() == static_cast<Eigen::Index>(N));
  for (size_t k = 0; k < N; k++)
    REQUIRE_THAT(n(k), WithinAbs(ref[k], tol));
  REQUIRE_THAT(n.sum(), WithinAbs(10.0, 1e-10));
}

} // namespace

TEST_CASE("CCSD density matches PySCF", "[cc][density]") {
  const auto r = occ::qm::cc::ccsd_density(rhf(), tight_all_electron());
  REQUIRE_THAT(r.e_corr, WithinAbs(-0.2133274269, 1e-8));
  REQUIRE_THAT(r.l1_norm, WithinAbs(0.0132555089, 1e-8));
  REQUIRE_THAT(r.l2_norm, WithinAbs(0.2125482514, 1e-8));
  check_occupations(r.occupations, ccsd_occ, 1e-8);
}

TEST_CASE("CCSD(T) density matches PySCF", "[cc][density]") {
  auto opts = tight_all_electron();
  opts.triples = true;
  const auto r = occ::qm::cc::ccsd_density(rhf(), opts);
  REQUIRE_THAT(r.e_triples, WithinAbs(-0.0030587074, 1e-9));
  REQUIRE_THAT(r.l1_norm, WithinAbs(0.0135673060, 1e-8));
  REQUIRE_THAT(r.l2_norm, WithinAbs(0.2202679718, 1e-8));
  check_occupations(r.occupations, ccsd_t_occ, 1e-8);
}

TEST_CASE("Relaxed MP2 density matches ORCA", "[cc][density][mp2]") {
  auto opts = tight_all_electron();
  const auto r = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  REQUIRE_THAT(r.e_corr, WithinAbs(-0.204003541, 1e-7));
  check_occupations(r.occupations, mp2_occ, 1e-6);
  // with the 1s frozen its occupation rises above 2, as in ORCA
  opts.n_frozen = 1;
  const auto f = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  REQUIRE_THAT(f.e_corr, WithinAbs(-0.201665956, 1e-7));
  REQUIRE_THAT(f.occupations(0), WithinAbs(2.00000262, 1e-6));
  REQUIRE(f.occupations(0) > 2.0);
}

TEST_CASE("Spanning local clusters reproduce the canonical density",
          "[cc][density][local]") {
  for (const bool triples : {false, true}) {
    auto opts = tight_all_electron();
    opts.triples = triples;
    opts.cluster_radius = 1e3;
    opts.domain_threshold = -1.0;
    opts.pair_threshold = -1.0;
    opts.lno_threshold = -1.0;
    opts.aux_basis = "def2-svp-rifit"; // screening only; it is off here
    const auto canonical = occ::qm::cc::ccsd_density(rhf(), opts);
    const auto local = occ::qm::cc::local_ccsd_density(rhf(), opts);
    REQUIRE(local.occupations.size() == canonical.occupations.size());
    for (Eigen::Index k = 0; k < canonical.occupations.size(); k++)
      REQUIRE_THAT(local.occupations(k),
                   WithinAbs(canonical.occupations(k), 1e-8));
    REQUIRE_THAT(local.e_corr, WithinAbs(canonical.e_corr, 1e-9));
  }
}

TEST_CASE("MP3 and MP4 energies match ORCA", "[cc][mpn]") {
  // ORCA's TightSCF orbitals move E2 and E3 by ~2e-8
  const auto all = occ::qm::cc::mpn_energies(
      occ::qm::cc::exact_eris(rhf().basis, rhf().mo, 0));
  REQUIRE_THAT(all.e2, WithinAbs(-0.204003540949, 1e-7));
  REQUIRE_THAT(all.e3, WithinAbs(-0.006789427245, 5e-8));
  REQUIRE_THAT(all.mp4_sdq() - all.mp3(), WithinAbs(-0.002229769988, 1e-8));
  REQUIRE_THAT(all.mp4() - all.mp3(), WithinAbs(-0.005233365840, 1e-8));
  const auto fc = occ::qm::cc::mpn_energies(
      occ::qm::cc::exact_eris(rhf().basis, rhf().mo, 1));
  REQUIRE_THAT(fc.e2, WithinAbs(-0.201665956431, 1e-7));
  REQUIRE_THAT(fc.e3, WithinAbs(-0.006997609515, 5e-8));
  REQUIRE_THAT(fc.mp4_sdq() - fc.mp3(), WithinAbs(-0.002257817888, 1e-8));
  REQUIRE_THAT(fc.mp4() - fc.mp3(), WithinAbs(-0.005236880343, 1e-8));
}
