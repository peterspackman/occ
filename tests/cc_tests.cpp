#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/gto/shell.h>
#include <occ/qm/cc/ccsd.h>
#include <occ/qm/cc/integrals.h>
#include <occ/qm/cc/laplace.h>
#include <occ/qm/cc/thc.h>
#include <occ/qm/cc/thc_mp2.h>
#include <occ/qm/correlation/mp2.h>
#include <occ/qm/cc/triples.h>
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/cc/uccsd_so.h>
#include <occ/qm/cc/uintegrals.h>
#include <occ/qm/hf.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf.h>
#include <vector>

using occ::Mat;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::cc::IsdfMethod;
using occ::qm::cc::ThcOptions;

namespace {

constexpr double ANG = occ::units::ANGSTROM_TO_BOHR;

struct Rhf {
  occ::gto::AOBasis basis;
  MolecularOrbitals mo;
  double e_hf{0.0};
};

Rhf run_rhf(const std::vector<occ::core::Atom> &atoms, const char *basis_name) {
  auto basis = occ::gto::AOBasis::load(atoms, basis_name);
  HartreeFock hf(basis);
  occ::qm::SCF<HartreeFock> scf(hf);
  double e = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  return {basis, scf.ctx.mo, e};
}

std::vector<occ::core::Atom> water() {
  return {
      {8, 0.0, 0.0, 0.0},
      {1, 0.0, 0.0, 0.96 * ANG},
      {1, 0.0, 0.93 * ANG, -0.24 * ANG},
  };
}

std::vector<occ::core::Atom> hydrogen_fluoride() {
  return {
      {1, 0.0, 0.0, 0.0},
      {9, 0.0, 0.0, 0.92 * ANG},
  };
}

std::vector<occ::core::Atom> oh_radical() {
  return {{8, 0.0, 0.0, 0.0}, {1, 0.0, 0.0, 0.97 * ANG}};
}

std::vector<occ::core::Atom> ch3_radical() {
  return {{6, 0.0, 0.0, 0.0},
          {1, 0.0, 0.0, 1.079 * ANG},
          {1, 0.934 * ANG, 0.0, -0.539 * ANG},
          {1, -0.934 * ANG, 0.0, -0.539 * ANG}};
}

std::vector<occ::core::Atom> o2_triplet() {
  return {{8, 0.0, 0.0, 0.0}, {8, 0.0, 0.0, 1.208 * ANG}};
}

struct Uhf {
  occ::gto::AOBasis basis;
  MolecularOrbitals mo;
  double e_hf{0.0};
};

Uhf run_uhf(const std::vector<occ::core::Atom> &atoms, const char *basis_name,
            int charge, int mult) {
  auto basis = occ::gto::AOBasis::load(atoms, basis_name);
  HartreeFock hf(basis);
  occ::qm::SCF<HartreeFock> scf(hf, occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(charge, mult);
  double e = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  return {basis, scf.ctx.mo, e};
}

} // namespace

TEST_CASE("Exact restricted CCSD vs PySCF", "[cc][ccsd]") {
  SECTION("H2O / STO-3G") {
    const auto scf = run_rhf(water(), "sto-3g");
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-74.9633586356, 1e-6));
    const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
    const auto res = occ::qm::cc::ccsd(eris);
    INFO("iters=" << res.iterations << " e_corr=" << res.e_corr);
    REQUIRE(res.converged);
    REQUIRE_THAT(res.e_corr, Catch::Matchers::WithinAbs(-0.0497527300, 1e-7));
  }
  SECTION("H2O / 6-31G") {
    const auto scf = run_rhf(water(), "6-31g");
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-75.9839234862, 1e-6));
    const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
    const auto res = occ::qm::cc::ccsd(eris);
    INFO("iters=" << res.iterations << " e_corr=" << res.e_corr);
    REQUIRE(res.converged);
    REQUIRE_THAT(res.e_corr, Catch::Matchers::WithinAbs(-0.1356128198, 1e-7));
  }
  SECTION("HF / STO-3G") {
    const auto scf = run_rhf(hydrogen_fluoride(), "sto-3g");
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-98.5711004441, 1e-6));
    const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
    const auto res = occ::qm::cc::ccsd(eris);
    INFO("iters=" << res.iterations << " e_corr=" << res.e_corr);
    REQUIRE(res.converged);
    REQUIRE_THAT(res.e_corr, Catch::Matchers::WithinAbs(-0.0260730845, 1e-7));
  }
}

TEST_CASE("Exact AO->MO: in-core packed and semidirect agree", "[cc][ccsd]") {
  // Default budget -> in-core packed transform; a 1-byte budget forces the
  // semidirect occ-blocked fallback. Both are exact and must agree.
  const auto scf = run_rhf(water(), "6-31g");
  const auto a = occ::qm::cc::ccsd(occ::qm::cc::exact_eris(scf.basis, scf.mo));
  const auto b = occ::qm::cc::ccsd(
      occ::qm::cc::exact_eris(scf.basis, scf.mo, /*n_frozen=*/0, /*budget=*/1));
  INFO("in-core=" << a.e_corr << "  semidirect=" << b.e_corr);
  REQUIRE_THAT(b.e_corr, Catch::Matchers::WithinAbs(a.e_corr, 1e-9));
}

TEST_CASE("Spin-orbital CCSD(T) == restricted (closed shell)", "[cc][uccsd]") {
  // The spin-orbital path on a closed-shell RHF reference must reproduce the
  // spin-adapted restricted CCSD(T) energies.
  const auto scf = run_rhf(water(), "sto-3g");
  const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
  const auto r = occ::qm::cc::ccsd(eris);
  const double r_t = occ::qm::cc::ccsd_t(r.t1, r.t2, eris);

  const auto u = occ::qm::cc::uccsd_so(scf.basis, scf.mo);
  INFO("restricted e_corr=" << r.e_corr << " e_(T)=" << r_t
                            << "  spinorbital e_corr=" << u.e_corr
                            << " e_(T)=" << u.e_triples);
  REQUIRE(u.converged);
  REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(r.e_corr, 1e-8));
  REQUIRE_THAT(u.e_triples, Catch::Matchers::WithinAbs(r_t, 1e-8));
}

TEST_CASE("Open-shell UHF CCSD(T) vs PySCF", "[cc][uccsd][uhf]") {
  // OH radical (doublet, 9 electrons) -- genuinely open-shell.
  const std::vector<occ::core::Atom> oh{{8, 0.0, 0.0, 0.0},
                                        {1, 0.0, 0.0, 0.97 * ANG}};
  auto basis = occ::gto::AOBasis::load(oh, "6-31g");
  occ::qm::HartreeFock hf(basis);
  occ::qm::SCF<occ::qm::HartreeFock> scf(hf,
                                         occ::qm::SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(0, 2);
  const double ehf = scf.compute_scf_energy();
  REQUIRE(scf.ctx.converged);
  REQUIRE(scf.ctx.mo.n_alpha != scf.ctx.mo.n_beta);
  REQUIRE_THAT(ehf, Catch::Matchers::WithinAbs(-75.3631682496, 1e-6));

  const auto u = occ::qm::cc::uccsd_so(basis, scf.ctx.mo, /*n_frozen=*/0);
  INFO("UHF e_corr=" << u.e_corr << " e_(T)=" << u.e_triples);
  REQUIRE(u.converged);
  REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0988276791, 1e-6));
  REQUIRE_THAT(u.e_triples, Catch::Matchers::WithinAbs(-0.0005574955, 1e-6));
}

TEST_CASE("Spin-adapted UCCSD: closed shell == restricted", "[cc][sauccsd]") {
  // On an RHF reference the spin-adapted unrestricted path must reproduce the
  // restricted CCSD correlation energy.
  const auto scf = run_rhf(water(), "6-31g");
  const auto r = occ::qm::cc::ccsd(occ::qm::cc::exact_eris(scf.basis, scf.mo));
  occ::qm::cc::UCCSDOptions opts;
  opts.with_triples = false;
  const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
  INFO("restricted e_corr=" << r.e_corr << "  uccsd e_corr=" << u.e_corr);
  REQUIRE(u.converged);
  REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(r.e_corr, 1e-8));
}

TEST_CASE("Spin-adapted UCCSD vs PySCF (open shell)", "[cc][sauccsd][uhf]") {
  occ::qm::cc::UCCSDOptions opts;
  opts.with_triples = false;

  SECTION("OH / 6-31g all-electron") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-75.3631682496, 1e-6));
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    // cross-check against the independent spin-orbital oracle too.
    const auto so = occ::qm::cc::uccsd_so(scf.basis, scf.mo, 0, false);
    INFO("uccsd e_corr=" << u.e_corr << "  uccsd_so e_corr=" << so.e_corr);
    REQUIRE(u.converged);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0988276791, 1e-7));
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(so.e_corr, 1e-8));
  }
  SECTION("OH / 6-31g frozen core") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    auto o = opts;
    o.n_frozen = 1;
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, o);
    REQUIRE(u.converged);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0979723461, 1e-7));
  }
  SECTION("CH3 / 6-31g all-electron") {
    const auto scf = run_uhf(ch3_radical(), "6-31g", 0, 2);
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-39.5465601969, 1e-6));
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    REQUIRE(u.converged);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0946540198, 1e-7));
  }
  SECTION("O2 / 6-31g triplet") {
    const auto scf = run_uhf(o2_triplet(), "6-31g", 0, 3);
    REQUIRE_THAT(scf.e_hf, Catch::Matchers::WithinAbs(-149.5455536710, 1e-6));
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    REQUIRE(u.converged);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.2351654873, 1e-7));
  }
}

TEST_CASE("Spin-adapted UCCSD(T) vs PySCF", "[cc][sauccsd][triples]") {
  occ::qm::cc::UCCSDOptions opts; // with_triples defaults true

  SECTION("OH / 6-31g all-electron") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    INFO("e_corr=" << u.e_corr << " (T)=" << u.e_triples);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0988276791, 1e-7));
    REQUIRE_THAT(u.e_triples, Catch::Matchers::WithinAbs(-0.0005574955, 1e-7));
  }
  SECTION("CH3 / 6-31g all-electron") {
    const auto scf = run_uhf(ch3_radical(), "6-31g", 0, 2);
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    INFO("e_corr=" << u.e_corr << " (T)=" << u.e_triples);
    REQUIRE_THAT(u.e_corr, Catch::Matchers::WithinAbs(-0.0946540198, 1e-7));
    REQUIRE_THAT(u.e_triples, Catch::Matchers::WithinAbs(-0.0010618842, 1e-7));
  }
  SECTION("OH / 6-31g frozen core") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    auto o = opts;
    o.n_frozen = 1;
    const auto u = occ::qm::cc::uccsd(scf.basis, scf.mo, o);
    INFO("e_corr=" << u.e_corr << " (T)=" << u.e_triples);
    REQUIRE_THAT(u.e_triples, Catch::Matchers::WithinAbs(-0.0005481171, 1e-7));
  }
}

TEST_CASE("Spin-adapted UCCSD DF backend vs exact", "[cc][sauccsd][df]") {
  occ::qm::cc::UCCSDOptions opts;
  opts.with_triples = false;

  SECTION("OH / 6-31g") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    const auto aux = occ::gto::AOBasis::load(oh_radical(), "def2-universal-jkfit");
    const auto ex = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    auto o = opts;
    o.backend = "df";
    const auto df = occ::qm::cc::uccsd(scf.basis, aux, scf.mo, o);
    INFO("exact=" << ex.e_corr << "  df=" << df.e_corr);
    REQUIRE(df.converged);
    REQUIRE_THAT(df.e_corr, Catch::Matchers::WithinAbs(ex.e_corr, 1e-3));
  }
  SECTION("CH3 / 6-31g") {
    const auto scf = run_uhf(ch3_radical(), "6-31g", 0, 2);
    const auto aux =
        occ::gto::AOBasis::load(ch3_radical(), "def2-universal-jkfit");
    const auto ex = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    auto o = opts;
    o.backend = "df";
    const auto df = occ::qm::cc::uccsd(scf.basis, aux, scf.mo, o);
    INFO("exact=" << ex.e_corr << "  df=" << df.e_corr);
    REQUIRE(df.converged);
    REQUIRE_THAT(df.e_corr, Catch::Matchers::WithinAbs(ex.e_corr, 1e-3));
  }
}

TEST_CASE("Spin-adapted UCCSD THC backend vs exact", "[cc][sauccsd][thc]") {
  occ::qm::cc::UCCSDOptions opts;
  opts.with_triples = false;

  SECTION("OH / 6-31g") {
    const auto scf = run_uhf(oh_radical(), "6-31g", 0, 2);
    const auto aux = occ::gto::AOBasis::load(oh_radical(), "def2-universal-jkfit");
    const auto ex = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    auto o = opts;
    o.backend = "thc";
    const auto th = occ::qm::cc::uccsd(scf.basis, aux, scf.mo, o);
    INFO("exact=" << ex.e_corr << "  thc=" << th.e_corr);
    REQUIRE(th.converged);
    REQUIRE_THAT(th.e_corr, Catch::Matchers::WithinAbs(ex.e_corr, 2e-3));
  }
  SECTION("CH3 / 6-31g") {
    const auto scf = run_uhf(ch3_radical(), "6-31g", 0, 2);
    const auto aux =
        occ::gto::AOBasis::load(ch3_radical(), "def2-universal-jkfit");
    const auto ex = occ::qm::cc::uccsd(scf.basis, scf.mo, opts);
    auto o = opts;
    o.backend = "thc";
    const auto th = occ::qm::cc::uccsd(scf.basis, aux, scf.mo, o);
    INFO("exact=" << ex.e_corr << "  thc=" << th.e_corr);
    REQUIRE(th.converged);
    REQUIRE_THAT(th.e_corr, Catch::Matchers::WithinAbs(ex.e_corr, 2e-3));
  }
}

TEST_CASE("Frozen-core CCSD(T) vs PySCF", "[cc][frozencore]") {
  // num_frozen_core: chemical core (C/N/O -> 1 orbital, H -> 0).
  REQUIRE(occ::qm::cc::num_frozen_core(
              occ::gto::AOBasis::load(water(), "6-31g")) == 1);
  REQUIRE(occ::qm::cc::num_frozen_core(
              occ::gto::AOBasis::load(hydrogen_fluoride(), "6-31g")) == 1);

  // Freeze the O 1s; compare to PySCF cc.CCSD(mf, frozen=1).
  const auto scf = run_rhf(water(), "6-31g");
  const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo, /*n_frozen=*/1);
  REQUIRE(eris.nocc == 4); // 5 occupied - 1 frozen
  const auto res = occ::qm::cc::ccsd(eris);
  const double et = occ::qm::cc::ccsd_t(res.t1, res.t2, eris);
  INFO("frozen-core e_corr=" << res.e_corr << " e_(T)=" << et);
  REQUIRE_THAT(res.e_corr, Catch::Matchers::WithinAbs(-0.1347050305, 1e-7));
  REQUIRE_THAT(et, Catch::Matchers::WithinAbs(-0.0009928025, 1e-7));

  // Frozen core must compose with the THC backend: the active-space THC result
  // tracks the exact frozen-core one within the THC tolerance.
  const auto aux = occ::gto::AOBasis::load(water(), "def2-universal-jkfit");
  occ::qm::cc::ThcOptions opts; // default c_isdf = 6 (rank capped at pair-space)
  const auto thc = occ::qm::cc::thc_eris(scf.basis, aux, scf.mo, opts,
                                         /*n_frozen=*/1);
  REQUIRE(thc.nocc == 4);
  const auto thc_res = occ::qm::cc::ccsd(thc);
  INFO("thc frozen-core e_corr=" << thc_res.e_corr);
  REQUIRE_THAT(thc_res.e_corr, Catch::Matchers::WithinAbs(res.e_corr, 1e-3));
}

TEST_CASE("Exact CCSD(T) triples vs PySCF", "[cc][triples]") {
  SECTION("H2O / STO-3G") {
    const auto scf = run_rhf(water(), "sto-3g");
    const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
    const auto res = occ::qm::cc::ccsd(eris);
    const double et = occ::qm::cc::ccsd_t(res.t1, res.t2, eris);
    INFO("e_(T)=" << et);
    REQUIRE_THAT(et, Catch::Matchers::WithinAbs(-0.0000678527, 1e-7));
  }
  SECTION("H2O / 6-31G") {
    const auto scf = run_rhf(water(), "6-31g");
    const auto eris = occ::qm::cc::exact_eris(scf.basis, scf.mo);
    const auto res = occ::qm::cc::ccsd(eris);
    const double et = occ::qm::cc::ccsd_t(res.t1, res.t2, eris);
    INFO("e_(T)=" << et);
    REQUIRE_THAT(et, Catch::Matchers::WithinAbs(-0.0010037679, 1e-6));
  }
}

TEST_CASE("THC CCSD(T) triples vs exact", "[cc][triples][thc]") {
  occ::qm::cc::ThcOptions opts;
  opts.c_isdf = 12.0;
  const auto atoms = water();
  const auto scf = run_rhf(atoms, "sto-3g");
  const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");

  const auto ex = occ::qm::cc::exact_eris(scf.basis, scf.mo);
  const auto exr = occ::qm::cc::ccsd(ex);
  const double et_exact = occ::qm::cc::ccsd_t(exr.t1, exr.t2, ex);

  const auto th = occ::qm::cc::thc_eris(scf.basis, aux, scf.mo, opts);
  const auto thr = occ::qm::cc::ccsd(th);
  const double et_thc = occ::qm::cc::ccsd_t(thr.t1, thr.t2, th);

  INFO("e_(T) exact=" << et_exact << "  thc=" << et_thc);
  REQUIRE_THAT(et_thc, Catch::Matchers::WithinAbs(et_exact, 1e-4));
}

TEST_CASE("DF restricted CCSD vs exact", "[cc][ccsd][df]") {
  const auto atoms = water();
  const auto scf = run_rhf(atoms, "6-31g");
  const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");

  const auto exact = occ::qm::cc::ccsd(occ::qm::cc::exact_eris(scf.basis, scf.mo));
  const auto df = occ::qm::cc::ccsd(occ::qm::cc::df_eris(scf.basis, aux, scf.mo));
  INFO("exact e_corr=" << exact.e_corr << "  df e_corr=" << df.e_corr);
  REQUIRE(df.converged);
  REQUIRE_THAT(df.e_corr, Catch::Matchers::WithinAbs(exact.e_corr, 1e-3));
}

namespace {

double rel_frob(const Eigen::Tensor<double, 4> &a,
                const Eigen::Tensor<double, 4> &b) {
  const Eigen::Tensor<double, 4> d = a - b;
  const Eigen::Tensor<double, 0> dn = d.square().sum();
  const Eigen::Tensor<double, 0> bn = b.square().sum();
  return std::sqrt(dn(0)) / std::sqrt(bn(0));
}

} // namespace

TEST_CASE("THC blocks track exact", "[cc][thc]") {
  const auto atoms = water();
  const auto scf = run_rhf(atoms, "sto-3g");
  const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");

  occ::qm::cc::ThcOptions opts;
  opts.c_isdf = 12.0;

  const auto ex = occ::qm::cc::exact_eris(scf.basis, scf.mo);
  const auto th = occ::qm::cc::thc_eris(scf.basis, aux, scf.mo, opts);

  const double ovvv_err = rel_frob(th.ovvv, ex.ovvv);
  INFO("ovvv relative error = " << ovvv_err);
  REQUIRE(ovvv_err < 1e-2);

  // random antisymmetric-free tau; compare the THC ladder to the exact one
  const int o = ex.nocc, v = ex.nvir;
  Eigen::Tensor<double, 4> tau(o, o, v, v);
  tau.setRandom();
  const double lad_err = rel_frob(th.ladder(tau), ex.ladder(tau));
  INFO("ladder relative error = " << lad_err);
  REQUIRE(lad_err < 1e-2);
}

TEST_CASE("THC restricted CCSD vs exact", "[cc][ccsd][thc]") {
  occ::qm::cc::ThcOptions opts;
  opts.c_isdf = 12.0;

  SECTION("H2O / STO-3G") {
    const auto atoms = water();
    const auto scf = run_rhf(atoms, "sto-3g");
    const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
    const auto exact = occ::qm::cc::ccsd(occ::qm::cc::exact_eris(scf.basis, scf.mo));
    const auto thc = occ::qm::cc::ccsd(occ::qm::cc::thc_eris(scf.basis, aux, scf.mo, opts));
    INFO("exact e_corr=" << exact.e_corr << "  thc e_corr=" << thc.e_corr);
    REQUIRE(thc.converged);
    REQUIRE_THAT(thc.e_corr, Catch::Matchers::WithinAbs(exact.e_corr, 1e-3));
  }
  SECTION("HF / STO-3G") {
    const auto atoms = hydrogen_fluoride();
    const auto scf = run_rhf(atoms, "sto-3g");
    const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");
    const auto exact = occ::qm::cc::ccsd(occ::qm::cc::exact_eris(scf.basis, scf.mo));
    const auto thc = occ::qm::cc::ccsd(occ::qm::cc::thc_eris(scf.basis, aux, scf.mo, opts));
    INFO("exact e_corr=" << exact.e_corr << "  thc e_corr=" << thc.e_corr);
    REQUIRE(thc.converged);
    REQUIRE_THAT(thc.e_corr, Catch::Matchers::WithinAbs(exact.e_corr, 1e-3));
  }
}

TEST_CASE("THC factorization reconstruction error", "[cc][thc]") {
  const auto atoms = water();
  const auto scf = run_rhf(atoms, "6-31g");
  const auto aux = occ::gto::AOBasis::load(atoms, "def2-universal-jkfit");

  SECTION("pivoted QR selector") {
    ThcOptions opts;
    opts.method = IsdfMethod::QR;
    opts.c_isdf = 12.0;
    const auto thc = occ::qm::cc::build_thc(scf.basis, aux, scf.mo, opts);
    const double err =
        occ::qm::cc::reconstruction_error(scf.basis, scf.mo, thc.X, thc.V);
    INFO("QR: n_isdf=" << thc.n_isdf << " cond=" << thc.metric_condition
                       << " kept=" << thc.metric_n_kept << " err=" << err);
    REQUIRE(thc.n_isdf > 0);
    REQUIRE(thc.metric_condition > 1e6); // LS-THC metric is ill-conditioned
    REQUIRE(err < 1e-2);
  }

  SECTION("pivoted Cholesky selector") {
    ThcOptions opts;
    opts.method = IsdfMethod::Cholesky;
    opts.c_isdf = 12.0;
    const auto thc = occ::qm::cc::build_thc(scf.basis, aux, scf.mo, opts);
    const double err =
        occ::qm::cc::reconstruction_error(scf.basis, scf.mo, thc.X, thc.V);
    INFO("Cholesky: n_isdf=" << thc.n_isdf << " cond=" << thc.metric_condition
                             << " kept=" << thc.metric_n_kept << " err=" << err);
    REQUIRE(thc.n_isdf > 0);
    REQUIRE(err < 1e-2);
  }
}

TEST_CASE("Laplace quadrature approximates 1/x", "[laplace]") {
  using occ::qm::cc::laplace_grid;
  using occ::qm::cc::laplace_max_rel_error;

  // The grid must reproduce 1/x across the whole range, with positive nodes and
  // weights (no cancellation). Tighter tolerance as the point count grows.
  SECTION("accuracy improves monotonically with point count (wide range)") {
    const double xmin = 0.5, xmax = 80.0; // stress: R = 160
    const double e8 = laplace_max_rel_error(laplace_grid(xmin, xmax, 8), xmin,
                                            xmax, 200);
    const double e12 = laplace_max_rel_error(laplace_grid(xmin, xmax, 12), xmin,
                                             xmax, 200);
    const double e16 = laplace_max_rel_error(laplace_grid(xmin, xmax, 16), xmin,
                                             xmax, 200);
    INFO("max rel err: n=8 " << e8 << "  n=12 " << e12 << "  n=16 " << e16);
    REQUIRE(e8 < 3e-2);
    REQUIRE(e12 < 2e-3);
    REQUIRE(e16 < 1e-4);
    REQUIRE(e12 < e8); // monotone improvement
    REQUIRE(e16 < e12);
  }

  SECTION("realistic gap range needs few points") {
    // A typical post-HF gap range (R ~ 25) is reproduced to well below the THC
    // factorisation error (~1e-3) with a modest point count.
    const double xmin = 0.8, xmax = 20.0;
    REQUIRE(laplace_max_rel_error(laplace_grid(xmin, xmax, 10), xmin, xmax,
                                  200) < 2e-3);
    REQUIRE(laplace_max_rel_error(laplace_grid(xmin, xmax, 13), xmin, xmax,
                                  200) < 1e-4);
  }

  SECTION("nodes are strictly positive and ordered") {
    const auto g = laplace_grid(0.3, 200.0, 13);
    REQUIRE(g.size() == 13);
    REQUIRE(g.points.minCoeff() > 0.0);
    for (int k = 1; k < g.size(); ++k)
      REQUIRE(g.points(k) > g.points(k - 1));
  }

  SECTION("pointwise reconstruction at sampled gaps") {
    const double xmin = 0.4, xmax = 50.0;
    const auto g = laplace_grid(xmin, xmax, 14);
    for (double x : {0.4, 1.0, 3.7, 12.0, 49.9}) {
      double approx = 0.0;
      for (int k = 0; k < g.size(); ++k)
        approx += g.weights(k) * std::exp(-x * g.points(k));
      REQUIRE_THAT(approx, Catch::Matchers::WithinRel(1.0 / x, 1e-4));
    }
  }
}

TEST_CASE("THC-MP2 restricted vs DF-MP2", "[thc][mp2]") {
  using Catch::Matchers::WithinAbs;
  const auto scf = run_rhf(water(), "def2-svp");
  const auto aux = occ::gto::AOBasis::load(water(), "def2-universal-jkfit");

  // DF-MP2 reference, all electrons correlated (disable energy-based freezing
  // so the active space matches THC-MP2's n_frozen=0).
  occ::qm::MP2 dfmp2(scf.basis, aux, scf.mo, scf.e_hf);
  dfmp2.set_algorithm(occ::qm::MP2::RI);
  dfmp2.set_orbital_energy_cutoffs(-1.0e9, 1.0e9);
  dfmp2.compute_correlation_energy();
  const auto &ref = dfmp2.results();

  occ::qm::cc::ThcMP2Options opts;
  opts.thc.c_isdf = 8.0;
  opts.n_laplace = 14;
  const auto thc = occ::qm::cc::thc_mp2(scf.basis, aux, scf.mo, opts);

  INFO("DF  os=" << ref.opposite_spin_correlation
                 << " ss=" << ref.same_spin_correlation
                 << " tot=" << ref.total_correlation);
  INFO("THC os=" << thc.opposite_spin << " ss=" << thc.same_spin
                 << " tot=" << thc.total << " n_isdf=" << thc.n_isdf
                 << " laplace_err=" << thc.laplace_max_rel_error);
  // The occ-virt-restricted core fit reproduces the ovov integrals MP2 needs
  // far more accurately than the all-pairs fit (water: ~1e-7 vs DF here).
  REQUIRE_THAT(thc.opposite_spin, WithinAbs(ref.opposite_spin_correlation, 1e-4));
  REQUIRE_THAT(thc.same_spin, WithinAbs(ref.same_spin_correlation, 1e-4));
  REQUIRE_THAT(thc.total, WithinAbs(ref.total_correlation, 1e-4));

  // opposite_spin_only (SOS fast path) must give the identical opposite-spin
  // energy and skip the same-spin exchange.
  occ::qm::cc::ThcMP2Options sos = opts;
  sos.opposite_spin_only = true;
  const auto thc_os = occ::qm::cc::thc_mp2(scf.basis, aux, scf.mo, sos);
  REQUIRE_THAT(thc_os.opposite_spin, WithinAbs(thc.opposite_spin, 1e-10));
  REQUIRE(thc_os.same_spin == 0.0);
  REQUIRE_THAT(thc_os.total, WithinAbs(thc.opposite_spin, 1e-10));
}

TEST_CASE("THC-MP2 unrestricted vs DF-UMP2", "[thc][mp2][uhf]") {
  using Catch::Matchers::WithinAbs;

  auto df_ump2 = [](const auto &scf, const occ::gto::AOBasis &aux) {
    occ::qm::MP2 m(scf.basis, aux, scf.mo, scf.e_hf);
    m.set_algorithm(occ::qm::MP2::RI);
    m.set_orbital_energy_cutoffs(-1.0e9, 1.0e9);
    m.compute_correlation_energy();
    return m.results();
  };

  SECTION("closed-shell UHF tracks DF-UMP2") {
    const auto scf = run_uhf(water(), "def2-svp", 0, 1);
    const auto aux = occ::gto::AOBasis::load(water(), "def2-universal-jkfit");
    const auto ref = df_ump2(scf, aux);

    occ::qm::cc::ThcMP2Options opts;
    opts.thc.c_isdf = 8.0;
    const auto thc = occ::qm::cc::thc_mp2(scf.basis, aux, scf.mo, opts);
    INFO("DF  os=" << ref.opposite_spin_correlation
                   << " ss=" << ref.same_spin_correlation
                   << " tot=" << ref.total_correlation);
    INFO("THC os=" << thc.opposite_spin << " ss=" << thc.same_spin
                   << " tot=" << thc.total);
    REQUIRE_THAT(thc.opposite_spin,
                 WithinAbs(ref.opposite_spin_correlation, 1e-3));
    REQUIRE_THAT(thc.same_spin, WithinAbs(ref.same_spin_correlation, 1e-3));
    REQUIRE_THAT(thc.total, WithinAbs(ref.total_correlation, 1e-3));
  }

  SECTION("open-shell doublet OH tracks DF-UMP2") {
    const auto scf = run_uhf(oh_radical(), "def2-svp", 0, 2);
    const auto aux = occ::gto::AOBasis::load(oh_radical(),
                                             "def2-universal-jkfit");
    const auto ref = df_ump2(scf, aux);

    occ::qm::cc::ThcMP2Options opts;
    opts.thc.c_isdf = 10.0;
    const auto thc = occ::qm::cc::thc_mp2(scf.basis, aux, scf.mo, opts);
    INFO("DF  os=" << ref.opposite_spin_correlation
                   << " ss=" << ref.same_spin_correlation
                   << " tot=" << ref.total_correlation);
    INFO("THC os=" << thc.opposite_spin << " ss=" << thc.same_spin
                   << " tot=" << thc.total << " n_isdf=" << thc.n_isdf);
    REQUIRE_THAT(thc.total, WithinAbs(ref.total_correlation, 2e-3));
  }
}
