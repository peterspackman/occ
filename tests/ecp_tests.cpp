// Native ECP integrals: special functions, convention checks independent of
// any other ECP code, internal consistency, independent references (mpmath,
// PySCF, brute-force quadrature) and finite differences.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <Eigen/Eigenvalues>
#include <array>
#include <Eigen/Cholesky>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <occ/gto/gto.h>
#include <occ/gto/shell.h>
#include <occ/ints/ecp.h>
#include <occ/qm/integral_engine.h>
#include "fd_test_utils.h"

using occ::Mat;
using occ::MatRM;
using occ::Vec3;
using occ::core::Atom;
using occ::gto::AOBasis;
using occ::ints::EcpBasisShell;
using occ::ints::EcpCentre;
using occ::ints::EcpEvaluator;
using occ::ints::EcpPrimitive;

namespace {

int ncart(int l) { return (l + 1) * (l + 2) / 2; }

std::vector<Atom> molecule(const std::string &name) {
  // bohr, deliberately low symmetry
  if (name == "HI")
    return {{53, 0.05, -0.02, 0.0}, {1, 0.31, 0.22, 3.02}};
  if (name == "HOI")
    return {{53, 0.0, 0.0, 0.0}, {8, 3.78, 0.05, 0.0}, {1, 4.35, 1.74, 0.12}};
  if (name == "CH2I2")
    return {{6, 0.0, 0.0, 0.0},       {53, 3.31, 2.35, 0.0},
            {53, -3.30, 2.36, 0.05},  {1, 0.05, -1.21, 1.68},
            {1, -0.02, -1.19, -1.70}};
  if (name == "AuClH")
    return {{79, 0.0, 0.0, 0.0}, {17, 0.3, 0.1, 4.2}, {1, -2.9, 0.4, -0.3}};
  if (name == "GdH2")
    return {{64, 0.0, 0.0, 0.1}, {1, 3.9, 0.2, 0.0}, {1, -1.4, 3.5, 0.3}};
  if (name == "GdF3")
    return {{64, 0.0, 0.0, 0.2},
            {9, 3.9, 0.0, 0.0},
            {9, -1.95, 3.38, 0.0},
            {9, -1.95, -3.38, 0.1}};
  throw std::runtime_error("unknown molecule");
}

Mat native_ecp_matrix(const AOBasis &basis) {
  EcpEvaluator ev(basis, 0);
  Mat V = Mat::Zero(basis.nbf(), basis.nbf());
  const auto &first = basis.first_bf();
  const bool sph = basis.is_pure();
  for (size_t c = 0; c < ev.centres().size(); c++)
    for (size_t p = 0; p < basis.size(); p++)
      for (size_t q = 0; q < basis.size(); q++) {
        Mat blk = ev.shell_pair(c, p, q);
        if (sph)
          blk = occ::gto::cartesian_to_spherical_transformation_matrix(
                    basis[p].l) *
                blk *
                occ::gto::cartesian_to_spherical_transformation_matrix(
                    basis[q].l)
                    .transpose();
        V.block(first[p], first[q], blk.rows(), blk.cols()) += blk;
      }
  return V;
}

} // namespace

TEST_CASE("Scaled modified spherical Bessel functions", "[ecp]") {
  // exp(-z) i_l(z), references from mpmath at 50 digits
  struct Ref {
    int l;
    double z, v;
  };
  const std::vector<Ref> refs{
      {0, 1e-10, 0.9999999999000000000066667},
      {3, 1e-3, 9.514291003175276395744494e-12},
      {0, 0.5, 0.6321205588285576784044762},
      {1, 0.5, 0.1036383235143269647865713},
      {5, 0.5, 0.000001840990395173499302609192},
      {0, 7.3, 0.06849311942790158969868336},
      {4, 7.3, 0.01655030967089430526774174},
      {12, 7.3, 0.000005089664478559882924484247},
      {12, 75, 0.00234567715700171269249947},
      {7, 1e4, 0.00004986018884258659382175607},
      {16, 30, 0.0001858674458541773040186849},
      {11, 0.02, 6.348027199385843170461708e-31},
      {9, 40, 0.004022471675757169723510742},
  };
  for (const auto &r : refs) {
    std::vector<double> out(r.l + 1);
    occ::ints::ecp_detail::scaled_bessel_i(r.l, r.z, out.data());
    INFO("l = " << r.l << " z = " << r.z);
    REQUIRE(out[r.l] == Catch::Approx(r.v).epsilon(1e-13));
  }
}

TEST_CASE("Boys and G_m functions", "[ecp]") {
  // F_m(T) = int_0^1 x^2m e^{-T x^2} dx and
  // G_m(T) = int_0^1 (1-v^2)^m e^{-T(1-v^2)} dv; references from mpmath
  // (incomplete gamma and 1F1(m+1; m+3/2; -T) closed forms)
  struct Ref {
    int m;
    double T, F, G;
  };
  const std::vector<Ref> refs{
      {0, 0, 1.0, 1.0},
      {1, 0, 0.33333333333333333333, 0.66666666666666666667},
      {5, 0, 0.090909090909090909091, 0.36940836940836940837},
      {12, 0, 0.04, 0.24816935117648538028},
      {19, 0, 0.025641025641025641026, 0.19940865344744050552},
      {0, 1e-6, 0.99999966666676666664, 0.99999933333359999992},
      {1, 1e-6, 0.33333313333340476189, 0.66666613333356190469},
      {5, 1e-6, 0.090909013986047319338, 0.36940802841618754574},
      {12, 1e-6, 0.039999962962980204337, 0.24816911219870702338},
      {19, 1e-6, 0.02564100125079336649, 0.19940845890250776226},
      {0, 0.3, 0.90839287703275053556, 0.82207285139752979993},
      {1, 0.3, 0.27929109391838778249, 0.52552760372674613314},
      {5, 0.3, 0.07058430316646883977, 0.28017497479965840495},
      {12, 0.3, 0.030305126728485953467, 0.18592335442768972137},
      {19, 0.3, 0.019277250746827951745, 0.14881792164358416959},
      {0, 4, 0.44104069538121083998, 0.15067019446189598302},
      {1, 4, 0.052840632061559582461, 0.044503968769632980894},
      {5, 4, 0.0036636899261127254897, 0.010187155730664546889},
      {12, 4, 0.0010298707317544743475, 0.0053960359587480913272},
      {19, 4, 0.0005819621860765623876, 0.0040669888297764780862},
      {0, 25, 0.17724538509027909508, 0.020426814884855367088},
      {1, 25, 0.0035449077015278230242, 0.00083535118255247442946},
      {5, 25, 5.3598970901605554938e-7, 2.8395295953429550444e-7},
      {12, 25, 2.2909578939562176727e-10, 2.4421421647087958081e-10},
      {19, 25, 6.7709602639771954242e-12, 1.5094336406949431333e-11},
      {0, 50, 0.12533141373155002512, 0.010103161564918598872},
      {1, 50, 0.0012533141373155002512, 0.00020419318056778486078},
      {5, 50, 1.1843818597631475259e-8, 4.0984410061095572737e-9},
      {12, 50, 3.9634072238087975146e-14, 2.2894964394146416276e-14},
      {19, 50, 1.0278168714781816057e-17, 8.3112436656023346669e-18},
      {0, 120, 0.080901079689820802252, 0.0041842494498401558795},
      {1, 120, 0.00033708783204092000938, 0.00003501715588115652895},
      {5, 120, 9.6012972993426109443e-11, 2.0619510921464034669e-11},
      {12, 120, 7.0053231740148050758e-19, 2.371580271802709724e-19},
      {19, 120, 3.9609253372900118464e-25, 1.7392540656016860537e-25},
      {0, 400, 0.044311346272637900682, 0.0012515683963201835973},
      {1, 400, 0.000055389182840797375853, 3.1328568155838268464e-6},
      {5, 400, 1.2778998482557011763e-13, 1.4759767253472608865e-14},
      {12, 400, 2.0391250471110024639e-25, 3.6284053150212456298e-26},
      {19, 400, 2.5215124866311924214e-35, 5.6757813831252688858e-36},
  };
  for (const auto &r : refs) {
    std::vector<double> F(20), G(20);
    occ::ints::ecp_detail::boys(19, r.T, F.data());
    occ::ints::ecp_detail::gfunc(19, r.T, G.data());
    INFO("m = " << r.m << " T = " << r.T);
    REQUIRE(F[r.m] == Catch::Approx(r.F).epsilon(1e-13));
    REQUIRE(G[r.m] == Catch::Approx(r.G).epsilon(1e-13));
  }
}

// A local-only r^0 ECP with a vanishing exponent is the identity operator,
// so its matrix must be occ's overlap matrix: checks the coefficient
// conventions (s/p factors, Cartesian normalisation) for s..g, Cartesian and
// spherical, with no other ECP code involved.
TEST_CASE("Local ECP with tiny exponent reproduces the overlap", "[ecp]") {
  for (bool spherical : {false, true}) {
    std::vector<Atom> atoms{{8, 0.0, 0.0, 0.0}, {9, 0.3, -0.4, 2.1}};
    auto basis = AOBasis::load(atoms, "def2-qzvp");
    basis.set_pure(spherical);
    int lmax = 0;
    for (const auto &sh : basis.shells())
      lmax = std::max(lmax, int(sh.l));
    REQUIRE(lmax >= 4);
    EcpCentre c;
    c.atom = 0;
    c.origin = Vec3(0.1, 0.2, 0.3);
    c.L = 0;
    c.local.push_back({2, 1e-15, 1.0});
    c.min_zeta = 1e-15;
    EcpEvaluator ev(occ::ints::ecp_shells_from_basis(basis), {c}, 0);
    occ::qm::IntegralEngine engine(basis);
    Mat S = engine.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);
    double maxerr = 0;
    const auto &first = basis.first_bf();
    for (size_t p = 0; p < basis.size(); p++) {
      for (size_t q = 0; q < basis.size(); q++) {
        MatRM v = ev.shell_pair(0, p, q);
        Mat blk = v;
        if (spherical) {
          blk = occ::gto::cartesian_to_spherical_transformation_matrix(
                    basis[p].l) *
                v *
                occ::gto::cartesian_to_spherical_transformation_matrix(
                    basis[q].l)
                    .transpose();
        }
        maxerr = std::max(maxerr,
                          (blk - S.block(first[p], first[q], blk.rows(),
                                         blk.cols()))
                              .cwiseAbs()
                              .maxCoeff());
      }
    }
    fmt::print("tiny local ECP vs overlap ({}): max |V - S| = {:.3e}\n",
               spherical ? "spherical" : "Cartesian", maxerr);
    REQUIRE(maxerr < 1e-12);
  }
}

// Semilocal projectors over l = 0..L-1 of the same radial function, plus a
// local term of minus that function, sum to minus the part of U beyond L-1.
// For shells on C with l < L that remainder vanishes exactly, and for nearby
// shells it decays rapidly with L. This checks the angular tables, the Bessel
// expansion and the radial quadrature against the analytic type-1 integral.
TEST_CASE("Semilocal projector completeness against analytic local term",
          "[ecp]") {
  std::vector<Atom> atoms{{8, 0.0, 0.0, 0.0}, {1, 0.25, -0.2, 0.3}};
  auto basis = AOBasis::load(atoms, "def2-tzvp");
  auto shells = occ::ints::ecp_shells_from_basis(basis);
  // l up to 3 on the O centre, H shells displaced a little
  double previous = 1.0;
  for (int L : {4, 8, 12, 16}) {
    EcpCentre c;
    c.atom = 0;
    c.origin = Vec3(0, 0, 0);
    c.L = L;
    c.semilocal.resize(L);
    const std::vector<EcpPrimitive> U{{2, 0.7, 1.3}, {2, 3.1, -0.4},
                                      {1, 1.7, 0.8}, {0, 2.2, 0.5}};
    for (int l = 0; l < L; l++)
      c.semilocal[l] = U;
    for (auto p : U) {
      p.coeff = -p.coeff;
      c.local.push_back(p);
    }
    c.min_zeta = 0.7;
    EcpEvaluator ev(shells, {c}, 0);
    double max_on = 0, max_off = 0;
    for (size_t p = 0; p < shells.size(); p++)
      for (size_t q = 0; q <= p; q++) {
        MatRM v = ev.shell_pair(0, p, q);
        const bool on = shells[p].atom == 0 && shells[q].atom == 0;
        if (on)
          max_on = std::max(max_on, v.cwiseAbs().maxCoeff());
        else
          max_off = std::max(max_off, v.cwiseAbs().maxCoeff());
      }
    fmt::print("projector completeness L = {:2d}: on-centre {:.3e}, "
               "off-centre {:.3e}\n",
               L, max_on, max_off);
    REQUIRE(max_on < 1e-12);
    // geometric decay of the remainder until it reaches round-off
    REQUIRE((max_off < 0.1 * previous || max_off < 1e-12));
    previous = max_off;
  }
}

// PySCF's ECP code (lib/gto/nr_ecp.c) is an independent implementation. These
// references were computed with PySCF 2.14 from the same basis and ECP data
// (occ's share/basis JSON converted to NWChem format), spherical functions:
// extreme generalised eigenvalues of (V, S) and Tr(S^-1 V), which do not
// depend on function ordering or phase conventions.
TEST_CASE("Native ECP matrix invariants match PySCF", "[ecp]") {
  struct Ref {
    std::string mol, basis;
    double lmin, lmax, trace;
  };
  const std::vector<Ref> refs{
      {"HI", "def2-tzvp", -4.695211285944077e-02, 2.421406450364605e+02,
       4.984698426716427e+02},
      {"CH2I2", "def2-svp", -1.863566833300742e-09, 1.140787744269182e+02,
       4.945155125285874e+02},
      {"AuClH", "def2-tzvp", 4.406254439716999e-20, 2.762137140095479e+02,
       4.176075425985196e+02},
      {"GdF3", "def2-svp", -1.674753745799748e+00, 6.138764364853982e+02,
       1.648225367403285e+03},
  };
  for (const auto &r : refs) {
    auto basis = AOBasis::load(molecule(r.mol), r.basis);
    basis.set_pure(true);
    occ::qm::IntegralEngine engine(basis);
    Mat S = engine.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);
    Mat V = native_ecp_matrix(basis);
    Eigen::GeneralizedSelfAdjointEigenSolver<Mat> es(V, S);
    const double trace = S.ldlt().solve(V).trace();
    const auto &ev = es.eigenvalues();
    fmt::print("{:6s} {:10s} lambda_min {:.3e} (ref {:.3e}) lambda_max "
               "diff {:.3e} trace diff {:.3e}\n",
               r.mol, r.basis, ev(0), r.lmin, ev(ev.size() - 1) - r.lmax,
               trace - r.trace);
    REQUIRE(std::abs(ev(0) - r.lmin) < 1e-9);
    REQUIRE(std::abs(ev(ev.size() - 1) - r.lmax) < 1e-10 * r.lmax);
    REQUIRE(std::abs(trace - r.trace) < 1e-10 * std::abs(r.trace));
  }
}

namespace {
// 4-point central difference of f(h) at h = 0
template <typename F> MatRM fd4(F &&f, double h) {
  return (-f(2 * h) + 8.0 * f(h) - 8.0 * f(-h) + f(-2 * h)) / (12.0 * h);
}

struct FdStats {
  double max_err{0}, max_val{0};
  void add(const MatRM &a, const MatRM &fd) {
    max_err = std::max(max_err, (a - fd).cwiseAbs().maxCoeff());
    max_val = std::max(max_val, fd.cwiseAbs().maxCoeff());
  }
};
} // namespace

// Integral-level finite differences of the native V with respect to each
// basis centre (C fixed) and of the derivative blocks for the second
// derivatives. Includes shells on the ECP centre and g functions on Gd.
TEST_CASE("Native ECP derivatives against finite differences", "[ecp]") {
  struct Case {
    std::string mol, basis;
    double h;
  };
  // Gd carries primitives with exponents ~7e4, which need a smaller step.
  for (const auto &cs : std::vector<Case>{{"HOI", "def2-svp", 2e-3},
                                          {"GdF3", "def2-svp", 2e-4}}) {
    auto basis = AOBasis::load(molecule(cs.mol), cs.basis);
    const auto shells = occ::ints::ecp_shells_from_basis(basis);
    const auto centres = occ::ints::ecp_centres_from_basis(basis);
    const double h = cs.h;
    FdStats s1, s2aa, s2ab, s2bb;
    EcpEvaluator ev(shells, centres, 2);
    // a spread of pairs: every shell against a few partners
    std::vector<std::pair<int, int>> pairs;
    for (int p = 0; p < int(shells.size()); p++)
      for (int q = 0; q <= p; q += 3)
        pairs.push_back({p, q});
    // Each centre moves along one generic unit direction u: the first
    // derivative along u, and the u-derivatives of all six analytic first
    // derivatives (AA, AB and BB second derivatives contracted with u).
    const Vec3 u = occ::test::generic_direction(3);
    // Just the pair's own shells, the moving one displaced: the pair's
    // integrals do not depend on the other shells.
    auto shifted = [&](int sa, int sb, int which, double d) {
      std::vector<EcpBasisShell> sh{shells[sa]};
      if (sb != sa)
        sh.push_back(shells[sb]);
      sh[which == 0 ? 0 : sh.size() - 1].origin += d * u;
      return sh;
    };
    auto sym = [](int x, int y) {
      return occ::ints::ecp_sym_index(std::min(x, y), std::max(x, y));
    };
    for (auto [p, q] : pairs) {
      auto first = ev.shell_pair_first_derivatives(0, p, q);
      std::array<MatRM, 6> aa, bb;
      std::array<MatRM, 9> ab;
      ev.shell_pair_second_derivatives(0, p, q, aa, ab, bb);
      for (int which = 0; which < 2; which++) {
        if (which == 1 && p == q)
          continue;
        MatRM fd = fd4(
            [&](double d) {
              EcpEvaluator e(shifted(p, q, which, d), centres, 0);
              return e.shell_pair(0, 0, p == q ? 0 : 1);
            },
            h);
        auto along_u = [&](auto &&term) {
          MatRM sum = u(0) * term(0);
          for (int x = 1; x < 3; x++)
            sum += u(x) * term(x);
          return sum;
        };
        if (p == q) {
          // moving the shell moves both A and B
          s1.add(along_u([&](int x) { return MatRM(first[x] + first[3 + x]); }),
                 fd);
          continue;
        }
        s1.add(along_u([&](int x) { return first[3 * which + x]; }), fd);
        // second derivatives: the same 4-point difference of all six
        // analytic first derivatives at once
        std::array<MatRM, 6> fd2;
        const double steps[4] = {-2 * h, -h, h, 2 * h};
        const double weights[4] = {1.0, -8.0, 8.0, -1.0};
        for (int s = 0; s < 4; s++) {
          EcpEvaluator e(shifted(p, q, which, steps[s]), centres, 1);
          const auto d1 = e.shell_pair_first_derivatives(0, 0, 1);
          for (int c = 0; c < 6; c++) {
            const MatRM term = weights[s] / (12.0 * h) * d1[c];
            fd2[c] = s == 0 ? term : MatRM(fd2[c] + term);
          }
        }
        for (int y = 0; y < 3; y++) {
          if (which == 0) {
            // d/du_A of dV/dA_y -> sum_x u_x AA(x, y); of dV/dB_y -> AB(x, y)
            s2aa.add(along_u([&](int x) { return aa[sym(x, y)]; }), fd2[y]);
            s2ab.add(along_u([&](int x) { return ab[3 * x + y]; }), fd2[3 + y]);
          } else {
            // d/du_B of dV/dB_y -> BB(x, y); of dV/dA_y -> AB(y, x)
            s2bb.add(along_u([&](int x) { return bb[sym(x, y)]; }), fd2[3 + y]);
            s2ab.add(along_u([&](int x) { return ab[3 * y + x]; }), fd2[y]);
          }
        }
      }
    }
    fmt::print("{} {}: first derivs max |an - fd| {:.2e} (max |fd| {:.2e}); "
               "AA {:.2e} ({:.2e}) AB {:.2e} ({:.2e}) BB {:.2e} ({:.2e})\n",
               cs.mol, cs.basis, s1.max_err, s1.max_val, s2aa.max_err,
               s2aa.max_val, s2ab.max_err, s2ab.max_val, s2bb.max_err,
               s2bb.max_val);
    REQUIRE(s1.max_err < 1e-8 * std::max(1.0, s1.max_val));
    REQUIRE(s2aa.max_err < 1e-6 * std::max(1.0, s2aa.max_val));
    REQUIRE(s2ab.max_err < 1e-6 * std::max(1.0, s2ab.max_val));
    REQUIRE(s2bb.max_err < 1e-6 * std::max(1.0, s2bb.max_val));
  }
}

// g functions on a light atom ~4 bohr from an iodine ECP (what the l+2
// shifted Hessian terms reach for d shells). Reference values from a
// brute-force radial x (theta, phi) product grid about the ECP centre with
// scipy's Y_lm, sharing no code with this implementation.
TEST_CASE("Native ECP g shell against brute-force quadrature", "[ecp]") {
  auto basis = AOBasis::load(molecule("CH2I2"), "def2-svp");
  auto shells = occ::ints::ecp_shells_from_basis(basis);
  auto centres = occ::ints::ecp_centres_from_basis(basis);
  REQUIRE(shells[5].l == 2);
  REQUIRE(shells[5].exponents.size() == 1);
  const double a = shells[5].exponents[0];
  shells[5].l = 4;
  shells[5].coeffs[0] *= a * a;
  EcpEvaluator ev(shells, centres, 0);
  MatRM v = ev.shell_pair(1, 26, 5);
  const double ref[4] = {5.8962169629e-07, -3.1047129476e-07,
                         -5.9839024741e-08, 2.0743412763e-07};
  for (int k = 0; k < 4; k++)
    REQUIRE(v(0, k) == Catch::Approx(ref[k]).epsilon(1e-9));
}

namespace {
using occ::test::displaced_along;

Mat random_symmetric(int n, unsigned seed) {
  std::srand(seed);
  Mat D = Mat::Random(n, n);
  return 0.5 * (D + D.transpose());
}
} // namespace

// Engine-level gradient of Tr(D V_ecp) with D held fixed, against finite
// differences of the engine's V, including h functions (def2-QZVPP Gd), so
// l + 1 = 6 once a derivative is taken.
TEST_CASE("ECP gradient kernel against finite differences", "[ecp]") {
  struct Case {
    std::string mol, basis;
    bool pure;
    double h;
  };
  for (const auto &cs : std::vector<Case>{{"HOI", "def2-svp", false, 1e-3},
                                          {"GdH2", "def2-qzvpp", true, 2e-4}}) {
    const auto atoms = molecule(cs.mol);
    auto load = [&](const std::vector<Atom> &a) {
      auto b = AOBasis::load(a, cs.basis);
      b.set_pure(cs.pure);
      return b;
    };
    auto basis = load(atoms);
    int lmax = 0;
    for (const auto &sh : basis.shells())
      lmax = std::max(lmax, int(sh.l));
    const Mat D = random_symmetric(basis.nbf(), 7);
    occ::qm::IntegralEngine engine(basis);
    const occ::Mat3N g = engine.effective_core_potential_gradient(D);
    // 4-point difference along a generic direction v over all coordinates
    const occ::Vec v = occ::test::generic_direction(atoms);
    const double h = cs.h;
    auto e = [&](double d) {
      occ::qm::IntegralEngine eng(load(displaced_along(atoms, v, d)));
      return D.cwiseProduct(eng.effective_core_potential()).sum();
    };
    const double fd =
        (-e(2 * h) + 8 * e(h) - 8 * e(-h) + e(-2 * h)) / (12 * h);
    const double err = std::abs(occ::test::flat(g).dot(v) - fd);
    fmt::print("{} {} (lmax {}): |grad.v - fd| {:.2e}, max |grad| {:.2e}, "
               "translational sum {:.2e}\n",
               cs.mol, cs.basis, lmax, err, g.cwiseAbs().maxCoeff(),
               g.rowwise().sum().cwiseAbs().maxCoeff());
    REQUIRE(err < 1e-7 * g.cwiseAbs().maxCoeff());
    REQUIRE(g.rowwise().sum().cwiseAbs().maxCoeff() <
            1e-11 * g.cwiseAbs().maxCoeff());
  }
}

// Skeleton Hessian sum P d2V/dRdR against finite differences of the analytic
// gradient (P fixed), for d shells next to iodine and g shells on Gd
// (l + 2 = 6). Also checks the derivative
// matrices: Tr(P dV/dR_k) must equal the gradient.
TEST_CASE("ECP Hessian and derivative-matrix kernels", "[ecp]") {
  struct Case {
    std::string mol, basis;
    bool pure;
    double h;
  };
  for (const auto &cs : std::vector<Case>{{"HOI", "def2-svp", true, 1e-3},
                                          {"HOI", "def2-svp", false, 1e-3},
                                          {"GdF3", "def2-svp", true, 2e-4}}) {
    const auto atoms = molecule(cs.mol);
    auto load = [&](const std::vector<Atom> &a) {
      auto b = AOBasis::load(a, cs.basis);
      b.set_pure(cs.pure);
      return b;
    };
    auto basis = load(atoms);
    const int n = atoms.size();
    const Mat P = random_symmetric(basis.nbf(), 11);
    occ::qm::IntegralEngine engine(basis);
    const Mat H = engine.effective_core_potential_hessian(P);
    // H v against a 4-point difference of the gradient along v
    const occ::Vec v = occ::test::generic_direction(atoms);
    const double h = cs.h;
    auto grad_at = [&](double d) {
      occ::qm::IntegralEngine eng(load(displaced_along(atoms, v, d)));
      return occ::test::flat(eng.effective_core_potential_gradient(P));
    };
    const occ::Vec fd = (-grad_at(2 * h) + 8 * grad_at(h) - 8 * grad_at(-h) +
                         grad_at(-2 * h)) /
                        (12 * h);
    const double scale = H.cwiseAbs().maxCoeff();
    const double err = (H * v - fd).cwiseAbs().maxCoeff();
    const double asym = (H - H.transpose()).cwiseAbs().maxCoeff();
    const double trans = occ::test::translational_sum_error(H);
    std::vector<Mat> H1(3 * n, Mat::Zero(basis.nbf(), basis.nbf()));
    engine.add_effective_core_potential_derivatives(H1);
    const occ::Mat3N g = engine.effective_core_potential_gradient(P);
    double h1err = 0;
    for (int a = 0; a < n; a++)
      for (int x = 0; x < 3; x++)
        h1err = std::max(h1err,
                         std::abs(P.cwiseProduct(H1[3 * a + x]).sum() - g(x, a)));
    fmt::print("{} {} {}: Hessian max |H v - fd| {:.2e} (max |H| {:.2e}), "
               "asymmetry {:.2e}, translational sum {:.2e}; "
               "|Tr(P H1) - grad| {:.2e}\n",
               cs.mol, cs.basis, cs.pure ? "spherical" : "Cartesian", err,
               scale, asym, trans, h1err);
    REQUIRE(err < 1e-7 * scale);
    REQUIRE(asym < 1e-11 * scale);
    REQUIRE(trans < 1e-11 * scale);
    REQUIRE(h1err < 1e-11 * g.cwiseAbs().maxCoeff());
  }
}
