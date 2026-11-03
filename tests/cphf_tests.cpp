#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fmt/core.h>
#include <occ/core/atom.h>
#include <occ/core/log.h>
#include <occ/qm/cphf.h>
#include <occ/qm/hf.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/scf.h>
#include "fd_test_utils.h"

using occ::Mat;
using occ::qm::HartreeFock;
using occ::qm::MolecularOrbitals;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;

namespace {

std::vector<occ::core::Atom> water() {
  // bohr
  return {{8, 0.0, 0.0, 0.2217561},
          {1, 0.0, 1.4309485, -0.8870244},
          {1, 0.0, -1.4309485, -0.8870244}};
}

// Density errors of ~1e-9 over the field steps stay far below the
// polarisability tolerances.
template <typename Proc>
double tight_scf(Proc &proc, const Mat &Vext = Mat(),
                 MolecularOrbitals *mo_out = nullptr) {
  SCF<Proc> scf(proc);
  scf.convergence_settings.energy_threshold = 1e-11;
  scf.convergence_settings.commutator_threshold = 1e-9;
  if (Vext.size() > 0)
    scf.set_external_potential(Vext, 0.0, "field");
  double e = scf.compute_scf_energy();
  if (mo_out)
    *mo_out = scf.molecular_orbitals();
  return e;
}

// Static polarisability from the coupled response to a uniform field. The
// field adds F . r to the electronic Hamiltonian, so the perturbation is r_x
// and alpha_xy = -d2E/dF_x dF_y = -2 n_occ sum_ai U^y_ai r^x_ai.
template <typename Proc>
occ::Mat3 cphf_polarisability(Proc &proc, const MolecularOrbitals &mo,
                              const occ::MatTriple &r) {
  occ::qm::cphf::Solver solver(mo, [&](const std::vector<Mat> &Cx) {
    return proc.compute_response_fock_occ(mo, Cx);
  });
  std::vector<Mat> B;
  for (int x = 0; x < 3; x++)
    B.push_back(-solver.vo_block(r[x]));
  const auto res = solver.solve(B);
  REQUIRE(res.converged);
  occ::Mat3 alpha;
  for (int x = 0; x < 3; x++)
    for (int y = 0; y < 3; y++)
      alpha(x, y) = -2.0 * solver.occupation() *
                    solver.vo_block(r[x]).cwiseProduct(res.U[y]).sum();
  return alpha;
}

// alpha f against the finite-field derivative -(mu(+h f) - mu(-h f)) / 2h of
// the electronic dipole, for one generic field direction f (the field adds
// h f . r); `scf_in_field(V)` converges the SCF with the external potential
// V and returns its orbitals. The tolerance is relative to |alpha f|; the
// O(h^2) truncation of the dipole difference is what limits it, hence the
// small field steps.
template <typename SCFInField>
void require_polarisability_matches_field(const std::string &label,
                                          const occ::Mat3 &alpha,
                                          const occ::MatTriple &r, double h,
                                          double rtol, SCFInField scf_in_field) {
  const occ::Vec f = occ::test::generic_direction(3);
  const Mat V = f(0) * r[0] + f(1) * r[1] + f(2) * r[2];
  const auto mp = scf_in_field(Mat(h * V)), mm = scf_in_field(Mat(-h * V));
  const occ::Vec fd = -(occ::test::electronic_dipole(mp, r) -
                        occ::test::electronic_dipole(mm, r)) /
                      (2 * h);
  const occ::Vec af = alpha * f;
  const double err = (af - fd).cwiseAbs().maxCoeff();
  fmt::print("{}: alpha f ({:.6f}, {:.6f}, {:.6f}), |alpha f - finite field| "
             "{:.2e}\n",
             label, af(0), af(1), af(2), err);
  REQUIRE(err < rtol * af.cwiseAbs().maxCoeff());
}

} // namespace

TEST_CASE("Response Fock is the linear part of the Fock build",
          "[cphf]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svp");
  HartreeFock hf(basis);
  MolecularOrbitals mo;
  tight_scf(hf, Mat(), &mo);
  // F2e[D] is linear in D, so G[D] must reproduce it.
  Mat F2e = hf.compute_fock(mo);
  Mat G = hf.compute_response_fock(mo, {mo.D})[0];
  REQUIRE((F2e - G).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("CPHF orbital Hessian is symmetric", "[cphf]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svp");
  HartreeFock hf(basis);
  MolecularOrbitals mo;
  tight_scf(hf, Mat(), &mo);
  occ::qm::cphf::Solver solver(mo, [&](const std::vector<Mat> &Cx) {
    return hf.compute_response_fock_occ(mo, Cx);
  });
  const auto &d = solver.orbital_energy_differences();
  Mat U1 = Mat::Random(d.rows(), d.cols());
  Mat U2 = Mat::Random(d.rows(), d.cols());
  auto AU = solver.apply({U1, U2});
  const double a = U2.cwiseProduct(AU[0]).sum();
  const double b = U1.cwiseProduct(AU[1]).sum();
  REQUIRE(a == Catch::Approx(b).epsilon(1e-10));
}

TEST_CASE("CPHF static polarisability matches finite field",
          "[cphf][polarisability]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svpd");
  HartreeFock hf(basis);
  MolecularOrbitals mo;
  tight_scf(hf, Mat(), &mo);
  occ::qm::IntegralEngine engine(basis);
  const auto r = engine.dipole_matrices();
  const occ::Mat3 alpha = cphf_polarisability(hf, mo, r);
  REQUIRE((alpha - alpha.transpose()).cwiseAbs().maxCoeff() < 1e-7);
  require_polarisability_matches_field("HF", alpha, r, 5e-4, 1e-4,
                                       [&](const Mat &V) {
                                         MolecularOrbitals m;
                                         tight_scf(hf, V, &m);
                                         return m;
                                       });
}

// COSX response: the shared multi-density sweep reproduces the SCF's own COSX
// exchange on the converged density, and a CPHF solved with it tracks the
// exact-exchange response to the accuracy of the grid.
TEST_CASE("COSX response Fock is the linear part of the COSX Fock build",
          "[cphf][cosx]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svp");
  HartreeFock hf(basis);
  hf.set_cosx_exchange(occ::numint::COSXGridLevel::Grid1);
  MolecularOrbitals mo;
  tight_scf(hf, Mat(), &mo);
  const Mat F2e = hf.compute_fock(mo);
  // Two densities in one sweep: the second is the first scaled, so its
  // response is too.
  auto G = hf.compute_response_fock(mo, {mo.D, Mat(-0.5 * mo.D)});
  fmt::print("COSX response vs Fock build: {:.2e}\n",
             (F2e - G[0]).cwiseAbs().maxCoeff());
  REQUIRE((F2e - G[0]).cwiseAbs().maxCoeff() < 1e-9);
  REQUIRE((G[1] + 0.5 * G[0]).cwiseAbs().maxCoeff() < 1e-12);
}

// A COSX SCF and its COSX response: the CPHF polarisability is the field
// derivative of that SCF's own dipole, which finite differences of the COSX
// SCF reproduce.
TEST_CASE("CPHF polarisability of a COSX SCF matches finite field",
          "[cphf][cosx][polarisability]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svpd");
  HartreeFock hf(basis);
  hf.set_cosx_exchange(occ::numint::COSXGridLevel::Grid1);
  MolecularOrbitals mo;
  tight_scf(hf, Mat(), &mo);
  occ::qm::IntegralEngine engine(basis);
  const auto r = engine.dipole_matrices();

  const occ::Mat3 alpha = cphf_polarisability(hf, mo, r);
  require_polarisability_matches_field("HF/COSX", alpha, r, 5e-4, 1e-5,
                                       [&](const Mat &V) {
                                         MolecularOrbitals m;
                                         tight_scf(hf, V, &m);
                                         return m;
                                       });
}

// ---------------------------------------------------------------------------
// Kohn-Sham response
// ---------------------------------------------------------------------------

#include <occ/dft/dft.h>

TEST_CASE("Kohn-Sham response Fock is the derivative of the Fock build",
          "[cphf][dft]") {
  occ::log::set_log_level(spdlog::level::warn);
  const std::string method =
      GENERATE(as<std::string>{}, "svwn5", "pbe", "b3lyp", "wb97x", "tpss",
               "r2scan", "m062x");
  auto basis = occ::gto::AOBasis::load(water(), "def2-svp");
  occ::dft::DFT dft(method, basis, occ::test::coarse_grid());
  occ::qm::SCF<occ::dft::DFT> scf(dft);
  scf.convergence_settings.energy_threshold = 1e-10;
  scf.compute_scf_energy();
  const auto mo = scf.molecular_orbitals();
  const int nbf = basis.nbf();
  Mat X = Mat::Random(nbf, nbf);
  const Mat D1 = 1e-2 * (X + X.transpose());
  const double eps = 1e-4;
  auto mp = mo, mm = mo;
  mp.D += eps * D1;
  mm.D -= eps * D1;
  const Mat fd = (dft.compute_fock(mp) - dft.compute_fock(mm)) / (2 * eps);
  const Mat G = dft.compute_response_fock(mo, {D1})[0];
  const double err = (G - fd).cwiseAbs().maxCoeff();
  fmt::print("{}: |response - FD of Fock| = {:.2e} (|G| {:.2e})\n", method, err,
             G.cwiseAbs().maxCoeff());
  REQUIRE(err < 1e-7 * std::max(1.0, G.cwiseAbs().maxCoeff()));
}

TEST_CASE("CPKS static polarisability matches finite field",
          "[cphf][dft][polarisability]") {
  occ::log::set_log_level(spdlog::level::warn);
  const std::string method =
      GENERATE(as<std::string>{}, "pbe", "hyb_mgga_xc_tpssh");
  auto basis = occ::gto::AOBasis::load(water(), "def2-svpd");
  occ::dft::DFT dft(method, basis, occ::test::coarse_grid());
  MolecularOrbitals mo;
  tight_scf(dft, Mat(), &mo);
  occ::qm::IntegralEngine engine(basis);
  const auto r = engine.dipole_matrices();
  const occ::Mat3 alpha = cphf_polarisability(dft, mo, r);
  require_polarisability_matches_field(method, alpha, r, 5e-4, 2e-4,
                                       [&](const Mat &V) {
                                         MolecularOrbitals m;
                                         tight_scf(dft, V, &m);
                                         return m;
                                       });
}

// The occupied-projected response (V1 C_o assembled point by point from the
// factorised density) against the full response contracted afterwards.
TEST_CASE("Projected Kohn-Sham response equals the full response times C_o",
          "[cphf][dft]") {
  occ::log::set_log_level(spdlog::level::warn);
  const std::string method =
      GENERATE(as<std::string>{}, "svwn5", "b3lyp", "wb97x", "tpss");
  // With diffuse functions, which def2-svp lacks.
  auto basis = occ::gto::AOBasis::load(water(), "def2-svpd");
  occ::dft::DFT dft(method, basis, occ::test::coarse_grid());
  MolecularOrbitals mo;
  tight_scf(dft, Mat(), &mo);
  const Eigen::Index nocc = mo.n_alpha;
  const Mat Co = mo.C.leftCols(nocc), Cv = mo.C.rightCols(mo.C.cols() - nocc);
  // Perturbed occupied coefficients: rotations into the virtuals, and one
  // occupied-occupied perturbation, as the Hessian's are.
  std::vector<Mat> Cx, D1;
  for (int k = 0; k < 3; k++)
    Cx.push_back(1e-2 * Cv * Mat::Random(Cv.cols(), nocc));
  const Mat M0 = Mat::Random(nocc, nocc);
  Cx.push_back(0.5 * Co * Mat(M0 + M0.transpose()));
  for (const auto &cx : Cx) {
    const Mat T = cx * Co.transpose();
    D1.push_back(T + T.transpose());
  }
  const auto Gp = dft.compute_response_fock_occ(mo, Cx);
  const auto G = dft.compute_response_fock(mo, D1);
  double err = 0.0, scale = 0.0;
  for (size_t k = 0; k < D1.size(); k++) {
    const Mat ref = G[k] * Co;
    err = std::max(err, (Gp[k] - ref).cwiseAbs().maxCoeff());
    scale = std::max(scale, ref.cwiseAbs().maxCoeff());
  }
  fmt::print("{}: |projected - full C_o| / |G C_o| = {:.2e}\n", method,
             err / scale);
  REQUIRE(err / scale < 1e-10);
}

// ---------------------------------------------------------------------------
// Unrestricted response
// ---------------------------------------------------------------------------

namespace {

// The water cation (a doublet with a non-degenerate SOMO, so its SCF
// converges cleanly), distorted so no block of the response vanishes by
// symmetry.
std::vector<occ::core::Atom> water_cation() {
  return occ::test::distorted_water();
}

template <typename Proc>
double open_shell_scf(Proc &proc, const Mat &Vext = Mat(),
                      MolecularOrbitals *mo_out = nullptr, int charge = 0) {
  SCF<Proc> scf(proc, SpinorbitalKind::Unrestricted);
  scf.set_charge_multiplicity(charge, 2);
  scf.convergence_settings.energy_threshold = 1e-11;
  scf.convergence_settings.commutator_threshold = 1e-9;
  if (Vext.size() > 0)
    scf.set_external_potential(Vext, 0.0, "field");
  double e = scf.compute_scf_energy();
  if (mo_out)
    *mo_out = scf.molecular_orbitals();
  return e;
}

// "method[/df|/cosx]": HF or a functional, with RI-JK or COSX exchange.
template <typename Proc> void configure(Proc &proc, const std::string &opt) {
  if (opt == "df")
    proc.set_density_fitting_basis("def2-universal-jkfit");
  else if (opt == "cosx") {
    proc.set_cosx_exchange(occ::numint::COSXGridLevel::Grid1);
    occ::qm::cosx::Settings cs;
    cs.f_threshold = 1e-14;
    proc.set_cosx_settings(cs);
  }
}

// The projected response G^s[D1] C_o^s against central differences of the
// Fock build with the occupied orbitals moved to C_o^s +/- eps Cx^s (RI-K
// builds from the orbitals, not D), which moves each spin density by
// +/- eps D1^s + eps^2 Cx^s Cx^s^T; the quadratic term cancels.
template <typename Proc>
double unrestricted_response_error(Proc &proc, const std::string &label) {
  MolecularOrbitals mo;
  open_shell_scf(proc, Mat(), &mo, 1);
  const auto channels = occ::qm::cphf::spin_channels(mo);
  const Eigen::Index nbf = mo.C.cols();
  const Mat Co = occ::qm::cphf::occupied_orbitals(mo);
  // Rotations into the virtuals and one occupied-occupied perturbation.
  std::vector<Mat> Cx;
  for (int k = 0; k < 3; k++) {
    Mat cx(nbf, Co.cols());
    Eigen::Index col = 0;
    for (const auto &c : channels) {
      const Eigen::Index no = c.Co.cols();
      if (k < 2) {
        cx.middleCols(col, no) = 1e-2 * c.Cv * Mat::Random(c.Cv.cols(), no);
      } else {
        const Mat M = Mat::Random(no, no);
        cx.middleCols(col, no) = 0.5 * c.Co * Mat(M + M.transpose());
      }
      col += no;
    }
    Cx.push_back(cx);
  }
  const auto G = proc.compute_response_fock_occ(mo, Cx);
  double err = 0.0, scale = 0.0;
  const double eps = 1e-4;
  for (size_t k = 0; k < Cx.size(); k++) {
    auto mp = mo, mm = mo;
    Eigen::Index col = 0;
    for (size_t s = 0; s < channels.size(); s++) {
      const Eigen::Index no = channels[s].Co.cols();
      const Mat d = eps * Cx[k].middleCols(col, no);
      mp.Cocc.block(s * nbf, 0, nbf, no) += d;
      mm.Cocc.block(s * nbf, 0, nbf, no) -= d;
      col += no;
    }
    mp.update_density_matrix();
    mm.update_density_matrix();
    const Mat fd = (proc.compute_fock(mp) - proc.compute_fock(mm)) / (2 * eps);
    const Mat ref = occ::qm::cphf::contract_occupied(fd, mo);
    err = std::max(err, (G[k] - ref).cwiseAbs().maxCoeff());
    scale = std::max(scale, ref.cwiseAbs().maxCoeff());
  }
  fmt::print("{}: |unrestricted response - FD of Fock| = {:.2e} (|G C_o| "
             "{:.2e})\n",
             label, err, scale);
  return err / std::max(1.0, scale);
}

} // namespace

TEST_CASE("Unrestricted response is the derivative of the Fock build",
          "[cphf][uhf]") {
  occ::log::set_log_level(spdlog::level::warn);
  const std::string method =
      GENERATE(as<std::string>{}, "hf", "hf/df", "hf/cosx", "svwn5", "b3lyp",
               "wb97x", "wb97x/df");
  const auto slash = method.find('/');
  const std::string name = method.substr(0, slash);
  const std::string opt =
      slash == std::string::npos ? "" : method.substr(slash + 1);
  auto basis = occ::gto::AOBasis::load(water_cation(), "def2-svp");
  double err;
  if (name == "hf") {
    HartreeFock hf(basis);
    configure(hf, opt);
    err = unrestricted_response_error(hf, method);
  } else {
    occ::dft::DFT dft(name, basis, occ::test::coarse_grid());
    configure(dft, opt);
    err = unrestricted_response_error(dft, method);
  }
  REQUIRE(err < 1e-7);
}

TEST_CASE("Unrestricted CPHF orbital Hessian is symmetric", "[cphf][uhf]") {
  occ::log::set_log_level(spdlog::level::warn);
  auto basis = occ::gto::AOBasis::load(water_cation(), "def2-svp");
  occ::dft::DFT dft("b3lyp", basis, occ::test::coarse_grid());
  MolecularOrbitals mo;
  open_shell_scf(dft, Mat(), &mo, 1);
  occ::qm::cphf::Solver solver(mo, [&](const std::vector<Mat> &Cx) {
    return dft.compute_response_fock_occ(mo, Cx);
  });
  Mat U1 = solver.vo_zero(), U2 = solver.vo_zero();
  U1.setRandom();
  U2.setRandom();
  auto AU = solver.apply({U1, U2});
  const double a = U2.cwiseProduct(AU[0]).sum();
  const double b = U1.cwiseProduct(AU[1]).sum();
  REQUIRE(a == Catch::Approx(b).epsilon(1e-10));
}

// Against the finite-field dipole derivative of the open-shell SCF, for the
// water cation (a non-degenerate doublet; OH's degenerate pi shell makes its
// finite-field UHF ill-conditioned).
TEST_CASE("Unrestricted CPHF/CPKS polarisability matches finite field",
          "[cphf][uhf][polarisability]") {
  occ::log::set_log_level(spdlog::level::warn);
  const std::string method =
      GENERATE(as<std::string>{}, "hf", "hf/df", "pbe", "b3lyp/df");
  const auto slash = method.find('/');
  const std::string name = method.substr(0, slash);
  const std::string opt =
      slash == std::string::npos ? "" : method.substr(slash + 1);
  auto basis = occ::gto::AOBasis::load(water(), "def2-svpd");
  occ::qm::IntegralEngine engine(basis);
  const auto r = engine.dipole_matrices();
  auto run = [&](auto &proc) {
    configure(proc, opt);
    MolecularOrbitals mo;
    open_shell_scf(proc, Mat(), &mo, 1);
    const occ::Mat3 alpha = cphf_polarisability(proc, mo, r);
    require_polarisability_matches_field(method, alpha, r, 5e-4, 2e-4,
                                         [&](const Mat &V) {
                                           MolecularOrbitals m;
                                           open_shell_scf(proc, V, &m, 1);
                                           return m;
                                         });
  };
  if (name == "hf") {
    HartreeFock hf(basis);
    run(hf);
  } else {
    occ::dft::DFT dft(name, basis, occ::test::coarse_grid());
    run(dft);
  }
}
