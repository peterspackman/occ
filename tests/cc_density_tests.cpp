#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <occ/core/units.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/density.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/correlation/mp2.h>
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
    REQUIRE_THAT(local.e_triples, WithinAbs(canonical.e_triples, 1e-9));
  }
}

TEST_CASE("DF densities track the exact backend", "[cc][density][df]") {
  auto opts = tight_all_electron();
  opts.tol = 1e-7;
  const auto ccsd = occ::qm::cc::ccsd_density(rhf(), opts);
  const auto mp2 = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  opts.backend = "df";
  opts.aux_basis = "def2-svp-rifit";
  const auto ccsd_df = occ::qm::cc::ccsd_density(rhf(), opts);
  const auto mp2_df = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  REQUIRE(ccsd_df.backend == "df");
  REQUIRE_THAT(ccsd_df.e_corr, WithinAbs(ccsd.e_corr, 2e-4));
  REQUIRE((ccsd_df.occupations - ccsd.occupations).cwiseAbs().maxCoeff() <
          1e-4);
  REQUIRE_THAT(mp2_df.e_corr, WithinAbs(mp2.e_corr, 1e-4));
  // the DF Lagrangian's energy is RI-MP2's on the same auxiliary basis
  auto aux = occ::gto::AOBasis::load(rhf().basis.atoms(), "def2-svp-rifit");
  aux.set_pure(rhf().basis.is_pure());
  occ::qm::MP2 ri(rhf().basis, aux, rhf().mo, rhf().energy.total);
  ri.set_algorithm(occ::qm::MP2::RI);
  ri.set_orbital_energy_cutoffs(-1e6, 1e6); // no frozen core
  REQUIRE_THAT(mp2_df.e_corr,
               WithinAbs(ri.compute_correlation_energy(), 1e-10));
  REQUIRE((mp2_df.occupations - mp2.occupations).cwiseAbs().maxCoeff() < 2e-5);
  // B_vv streamed one row of a at a time gives the same density
  auto tiny = opts;
  tiny.max_memory_gb = 1e-9;
  const auto mp2_tiny = occ::qm::cc::mp2_relaxed_density(rhf(), tiny);
  REQUIRE_THAT(mp2_tiny.e_corr, WithinAbs(mp2_df.e_corr, 1e-12));
  REQUIRE((mp2_tiny.occupations - mp2_df.occupations).cwiseAbs().maxCoeff() <
          1e-10);
  // frozen core: the frozen-core block of P and its response
  opts.n_frozen = 1;
  const auto fc_df = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  opts.backend = "exact";
  const auto fc = occ::qm::cc::mp2_relaxed_density(rhf(), opts);
  REQUIRE_THAT(fc_df.e_corr, WithinAbs(fc.e_corr, 1e-4));
  REQUIRE((fc_df.occupations - fc.occupations).cwiseAbs().maxCoeff() < 2e-5);
  REQUIRE(fc_df.occupations(0) > 2.0);
}

TEST_CASE("Truncated local clusters approach the canonical density",
          "[cc][density][local]") {
  // water dimer cc-pVDZ: radius 1 A gives one cluster per monomer, 3 A lets
  // each cluster take in most of the other monomer
  constexpr double A = occ::units::ANGSTROM_TO_BOHR;
  const std::vector<occ::core::Atom> atoms{
      {8, -1.551007 * A, -0.114520 * A, 0.0},
      {1, -1.934259 * A, 0.762503 * A, 0.0},
      {1, -0.599677 * A, 0.040712 * A, 0.0},
      {8, 1.350625 * A, 0.111469 * A, 0.0},
      {1, 1.680398 * A, -0.373741 * A, -0.758561 * A},
      {1, 1.680398 * A, -0.373741 * A, 0.758561 * A}};
  auto basis = occ::gto::AOBasis::load(atoms, "cc-pvdz");
  basis.set_pure(true);
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf);
  scf.convergence_settings.energy_threshold = 1e-11;
  scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  const auto wfn = scf.wavefunction();

  CorrelatedDensityOptions opts;
  opts.n_frozen = 0;
  opts.tol = 1e-7;
  opts.triples = true;
  opts.aux_basis = "def2-svp-rifit";
  const auto canonical = occ::qm::cc::ccsd_density(wfn, opts);
  auto local_at = [&](const double radius) {
    opts.cluster_radius = radius;
    const auto r = occ::qm::cc::local_ccsd_density(wfn, opts);
    REQUIRE(r.n_lmo == 10);
    REQUIRE(r.n_clusters == 2);
    REQUIRE(r.max_cluster_occ < r.n_lmo);         // truncated occupied space
    REQUIRE(r.max_cluster_vir < canonical.n_vir); // and virtual space
    REQUIRE_THAT(r.occupations.sum(), WithinAbs(20.0, 1e-10));
    return r;
  };
  const auto near = local_at(1.0), far = local_at(3.0);
  auto max_dn = [&](const occ::qm::cc::CorrelatedDensity &r) {
    return (r.occupations - canonical.occupations).cwiseAbs().maxCoeff();
  };
  // 99.5 % and 99.9 % of the CCSD correlation energy; (T) partitioned by
  // central LMO converges with the cluster too
  REQUIRE(near.e_corr / canonical.e_corr > 0.994);
  REQUIRE(far.e_corr / canonical.e_corr > 0.999);
  REQUIRE(std::abs(far.e_triples - canonical.e_triples) <
          std::abs(near.e_triples - canonical.e_triples));
  REQUIRE_THAT(far.e_triples, WithinAbs(canonical.e_triples, 2e-4));
  REQUIRE(max_dn(near) < 1e-3);
  REQUIRE(max_dn(far) < max_dn(near));
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

namespace {

// ORCA 6.1.0 AutoCI unrelaxed densities (%autoci density unrelaxed)
constexpr std::array<double, 24> mp3_occ{
    1.999915773813, 1.985661784135, 1.973702071950, 1.968648639902,
    1.966229357415, 0.024377209706, 0.022180608905, 0.016239111965,
    0.010783938325, 0.006074443783, 0.005773204255, 0.005311825070,
    0.004663934998, 0.004554560921, 0.001125255748, 0.001125093303,
    0.000734889516, 0.000664915046, 0.000587592294, 0.000548531000,
    0.000513285935, 0.000490790207, 0.000052501593, 0.000040680215};

constexpr std::array<double, 24> mp4_sdq_occ{
    1.999915945826, 1.984711924471, 1.973070814366, 1.967314715391,
    1.964745794384, 0.025970440535, 0.023919231243, 0.016933633869,
    0.011456444716, 0.005959616296, 0.005679385498, 0.005147216186,
    0.004537916329, 0.004426693801, 0.001213224596, 0.001152612750,
    0.000741770946, 0.000740148418, 0.000675373039, 0.000555525988,
    0.000522202536, 0.000504326705, 0.000060552809, 0.000044489303};

constexpr std::array<double, 24> mp4_occ{
    1.999914406247, 1.984274300365, 1.971968172897, 1.965625772591,
    1.962719664667, 0.027364938764, 0.025005173092, 0.017577711958,
    0.011817408725, 0.006274482577, 0.005958985372, 0.005377166917,
    0.004664419368, 0.004547821106, 0.001349431881, 0.001250163400,
    0.000858402579, 0.000817711751, 0.000782625933, 0.000609186636,
    0.000558469763, 0.000558076384, 0.000071783877, 0.000053723150};

// singular values of the ov block in the canonical MOs, invariant to occupied
// and virtual rotations (so to MO phases)
occ::Vec ov_singular_values(const occ::qm::cc::CorrelatedDensity &r) {
  const occ::Mat Ci = rhf().mo.C.inverse();
  const occ::Mat P = 2.0 * Ci * r.natural.mo.D * Ci.transpose();
  return Eigen::JacobiSVD<occ::Mat>(P.topRightCorner(5, 19)).singularValues();
}

void check_ov(const occ::qm::cc::CorrelatedDensity &r,
              const std::array<double, 5> &ref) {
  const occ::Vec sv = ov_singular_values(r);
  for (size_t k = 0; k < ref.size(); k++)
    REQUIRE_THAT(sv(k), WithinAbs(ref[k], 2e-9));
}

} // namespace

TEST_CASE("MP3 and MP4 densities match ORCA", "[cc][density][mpn]") {
  constexpr double e2 = -0.204003540949, e3 = -0.006789427245;
  auto opts = tight_all_electron();
  const auto mp3 = occ::qm::cc::mpn_density(rhf(), 3, opts);
  REQUIRE_THAT(mp3.e_corr, WithinAbs(e2 + e3, 1e-7));
  check_occupations(mp3.occupations, mp3_occ, 2e-7);
  REQUIRE(ov_singular_values(mp3).maxCoeff() < 1e-12); // no ov block
  const auto sdq = occ::qm::cc::mpn_density(rhf(), 4, opts);
  REQUIRE_THAT(sdq.e_corr, WithinAbs(e2 + e3 - 0.002229769988, 1e-7));
  check_occupations(sdq.occupations, mp4_sdq_occ, 2e-7);
  check_ov(sdq, {4.050671e-05, 3.225539e-05, 2.398397e-05, 9.162272e-06,
                 1.977190e-06});
  opts.triples = true;
  const auto mp4 = occ::qm::cc::mpn_density(rhf(), 4, opts);
  REQUIRE_THAT(mp4.e_corr, WithinAbs(e2 + e3 - 0.005233365840, 1e-7));
  check_occupations(mp4.occupations, mp4_occ, 2e-7);
  check_ov(mp4, {6.326736e-05, 6.116768e-05, 3.684343e-05, 1.884559e-05,
                 2.634308e-06});
}
