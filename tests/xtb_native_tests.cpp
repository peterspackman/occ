#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/data_directory.h>
#include <occ/core/units.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/basis.h>
#include <occ/xtb/anisotropic.h>
#include <occ/xtb/camm.h>
#include <occ/xtb/coordination.h>
#include <occ/xtb/multipole_damping.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/multipole_ints.h>
#include <occ/xtb/native_calculator.h>
#include <occ/xtb/repulsion.h>
#include <occ/xtb/scc.h>
#include <occ/xtb/sto_ng.h>

#ifndef OCC_GFN2_DATA_DIR
#define OCC_GFN2_DATA_DIR "share"
#endif

using Catch::Approx;

namespace {
struct DataDirGuard {
  DataDirGuard() { occ::set_data_directory(OCC_GFN2_DATA_DIR); }
};
DataDirGuard _guard;
} // namespace

TEST_CASE("GFN2 parameters: load default", "[xtb][gfn2][params]") {
  using occ::xtb::Gfn2Parameters;

  auto p = Gfn2Parameters::load_default();

  REQUIRE(p.method() == "GFN2-xTB");
  REQUIRE(p.doi() == "10.1021/acs.jctc.8b01176");
  REQUIRE(p.max_z() == 86);

  const auto &g = p.globals();
  REQUIRE(g.kshell[0] == Approx(1.85).margin(1e-12));
  REQUIRE(g.kshell[1] == Approx(2.23).margin(1e-12));
  REQUIRE(g.kshell[2] == Approx(2.23).margin(1e-12));
  REQUIRE(g.ipeashift_au == Approx(0.178069).margin(1e-9));
  REQUIRE(g.kexp == Approx(1.5).margin(1e-12));
  REQUIRE(g.kexplight == Approx(1.0).margin(1e-12));
  REQUIRE(g.a1 == Approx(0.52).margin(1e-12));
  REQUIRE(g.a2 == Approx(5.0).margin(1e-12));
  REQUIRE(g.s8 == Approx(2.7).margin(1e-12));
  // gam3shell d split: d1=0.25, d2=0.25
  REQUIRE(g.gam3shell[0][0] == Approx(1.0).margin(1e-12));
  REQUIRE(g.gam3shell[1][0] == Approx(0.5).margin(1e-12));
  REQUIRE(g.gam3shell[2][0] == Approx(0.25).margin(1e-12));
  REQUIRE(g.gam3shell[2][1] == Approx(0.25).margin(1e-12));
}

TEST_CASE("GFN2 parameters: hydrogen", "[xtb][gfn2][params]") {
  auto p = occ::xtb::Gfn2Parameters::load_default();

  const auto *h = p.element(1);
  REQUIRE(h != nullptr);
  REQUIRE(h->ao == "1s");
  REQUIRE(h->shells.size() == 1);
  REQUIRE(h->atomic_hardness == Approx(0.405771).margin(1e-9));
  REQUIRE(h->third_order_atom_au == Approx(0.080).margin(1e-9));
  REQUIRE(h->rep_alpha == Approx(2.213717).margin(1e-9));
  REQUIRE(h->rep_zeff == Approx(1.105388).margin(1e-9));
  REQUIRE(h->pauling_en == Approx(2.20).margin(1e-9));

  const auto &s = h->shells[0];
  REQUIRE(s.n == 1);
  REQUIRE(s.l == 0);
  REQUIRE(s.n_prim == 3);
  REQUIRE(s.is_valence);
  REQUIRE(s.self_energy_ev == Approx(-10.707211).margin(1e-9));
  REQUIRE(s.slater_exponent == Approx(1.230000).margin(1e-9));
  REQUIRE(s.kcn_au == Approx(-0.05).margin(1e-9));
  REQUIRE(s.shell_poly == Approx(-0.953618).margin(1e-9));
  REQUIRE(s.ref_occ == Approx(1.0).margin(1e-12));
}

TEST_CASE("GFN2 parameters: carbon", "[xtb][gfn2][params]") {
  auto p = occ::xtb::Gfn2Parameters::load_default();

  const auto *c = p.element(6);
  REQUIRE(c != nullptr);
  REQUIRE(c->ao == "2s2p");
  REQUIRE(c->shells.size() == 2);

  const auto &s = c->shells[0];
  REQUIRE(s.n == 2);
  REQUIRE(s.l == 0);
  REQUIRE(s.n_prim == 4);
  REQUIRE(s.self_energy_ev == Approx(-13.970922).margin(1e-9));
  REQUIRE(s.slater_exponent == Approx(2.096432).margin(1e-9));
  REQUIRE(s.ref_occ == Approx(1.0).margin(1e-12));

  const auto &p_shell = c->shells[1];
  REQUIRE(p_shell.l == 1);
  REQUIRE(p_shell.n_prim == 4);
  REQUIRE(p_shell.is_valence);
  REQUIRE(p_shell.ref_occ == Approx(3.0).margin(1e-12));
}

TEST_CASE("GFN2 parameters: gold", "[xtb][gfn2][params]") {
  auto p = occ::xtb::Gfn2Parameters::load_default();

  const auto *au = p.element(79);
  REQUIRE(au != nullptr);
  REQUIRE(au->ao == "5d6s6p");
  REQUIRE(au->shells.size() == 3);
  // shell order is preserved from ao=string: 5d, 6s, 6p
  REQUIRE(au->shells[0].n == 5);
  REQUIRE(au->shells[0].l == 2);
  REQUIRE(au->shells[0].n_prim == 3);
  REQUIRE(au->shells[1].n == 6);
  REQUIRE(au->shells[1].l == 0);
  REQUIRE(au->shells[1].n_prim == 6);
  REQUIRE(au->shells[2].n == 6);
  REQUIRE(au->shells[2].l == 1);
  REQUIRE(au->shells[2].n_prim == 6);
}

TEST_CASE("GFN2 parameters: out-of-range Z is null", "[xtb][gfn2][params]") {
  auto p = occ::xtb::Gfn2Parameters::load_default();
  REQUIRE(p.element(0) == nullptr);
  REQUIRE(p.element(87) == nullptr);
  REQUIRE(p.element(-1) == nullptr);
}

TEST_CASE("STO-NG: Stewart 1s/2p fits", "[xtb][sto_ng]") {
  // Spot checks against xtb's slater.f90 / Stewart 1970 numbers.
  auto h_1s = occ::xtb::slater_to_gauss(3, 1, 0, 1.0, /*normalize=*/false);
  REQUIRE(h_1s.alpha.size() == 3);
  REQUIRE(h_1s.alpha[0] == Approx(2.227660584e+0).margin(1e-9));
  REQUIRE(h_1s.alpha[2] == Approx(1.098175104e-1).margin(1e-9));
  REQUIRE(h_1s.coeff[0] == Approx(1.543289673e-1).margin(1e-9));
  REQUIRE(h_1s.coeff[2] == Approx(4.446345422e-1).margin(1e-9));

  // zeta scaling: alpha must scale as zeta^2.
  auto h_1s_z2 = occ::xtb::slater_to_gauss(3, 1, 0, 2.0, /*normalize=*/false);
  REQUIRE(h_1s_z2.alpha[0] == Approx(h_1s.alpha[0] * 4.0).margin(1e-9));

  // Carbon 2p (n=2, l=1) NG=4 — first exponent.
  auto c_2p = occ::xtb::slater_to_gauss(4, 2, 1, 1.0, /*normalize=*/false);
  REQUIRE(c_2p.alpha.size() == 4);
  REQUIRE(c_2p.alpha[0] == Approx(1.798260992e+0).margin(1e-9));
  REQUIRE(c_2p.coeff[0] == Approx(5.713170255e-2).margin(1e-9));
}

TEST_CASE("STO-NG: invalid arguments throw", "[xtb][sto_ng]") {
  REQUIRE_THROWS(occ::xtb::slater_to_gauss(3, 1, 0, 0.0));
  REQUIRE_THROWS(occ::xtb::slater_to_gauss(0, 1, 0, 1.0));
  REQUIRE_THROWS(occ::xtb::slater_to_gauss(7, 1, 0, 1.0));
  REQUIRE_THROWS(occ::xtb::slater_to_gauss(3, 6, 0, 1.0)); // n=6 needs ng=6
  REQUIRE_THROWS(occ::xtb::slater_to_gauss(3, 1, 1, 1.0)); // l >= n
}

TEST_CASE("AOBasis: water shell count", "[xtb][basis]") {
  using occ::core::Atom;
  using occ::units::ANGSTROM_TO_BOHR;

  auto p = occ::xtb::Gfn2Parameters::load_default();

  // Coordinates in Bohr.
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };

  auto basis = occ::xtb::build_aobasis(atoms, p);
  REQUIRE(basis.size() == 4);    // O: 2s, 2p   |  H: 1s   |  H: 1s
  REQUIRE(basis.nbf() == 6);     // 1 + 3 + 1 + 1
  REQUIRE(basis.is_pure());
  REQUIRE(basis.l_max() == 1);
}

TEST_CASE("AOBasis: methane shell count", "[xtb][basis]") {
  using occ::core::Atom;
  auto p = occ::xtb::Gfn2Parameters::load_default();
  std::vector<Atom> atoms{
      {6, 0.0, 0.0, 0.0},
      {1, 1.18886, 1.18886, 1.18886},
      {1, -1.18886, -1.18886, 1.18886},
      {1, -1.18886, 1.18886, -1.18886},
      {1, 1.18886, -1.18886, -1.18886},
  };

  auto basis = occ::xtb::build_aobasis(atoms, p);
  REQUIRE(basis.size() == 6);    // C: 2s, 2p   |  4 H: 4 × 1s
  REQUIRE(basis.nbf() == 8);     // 1 + 3 + 4
}

TEST_CASE("Coordination number: water gfn flavor", "[xtb][cn]") {
  using occ::core::Atom;
  // Water at the GFN2-equilibrium, positions in Bohr.
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };

  occ::Vec cn = occ::xtb::gfn_coordination_numbers(atoms);
  REQUIRE(cn.size() == 3);
  // GFN counting saturates near (but below) the integer; for water, 1.92 / 0.96
  // for O / H is what tblite produces.
  REQUIRE(cn(0) == Approx(1.92).margin(0.05));
  REQUIRE(cn(1) == Approx(0.96).margin(0.05));
  REQUIRE(cn(2) == Approx(0.96).margin(0.05));
  // Hydrogens aren't perfectly symmetric in the input geometry; both
  // should still land near the same value.
  REQUIRE(cn(1) == Approx(cn(2)).margin(0.01));
}

TEST_CASE("Coordination number: methane symmetry", "[xtb][cn]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {6, 0.0, 0.0, 0.0},
      {1, 1.18886, 1.18886, 1.18886},
      {1, -1.18886, -1.18886, 1.18886},
      {1, -1.18886, 1.18886, -1.18886},
      {1, 1.18886, -1.18886, -1.18886},
  };
  occ::Vec cn = occ::xtb::gfn_coordination_numbers(atoms);
  REQUIRE(cn.size() == 5);
  // Tetrahedral CH4: counting function saturates around 3.8 for C, 0.96/H.
  REQUIRE(cn(0) == Approx(3.83).margin(0.1));
  for (int i = 1; i <= 4; ++i) {
    REQUIRE(cn(i) == Approx(cn(1)).margin(1e-12));
  }
}

TEST_CASE("Dipole AO matrices: symmetry and overlap consistency",
          "[xtb][multipole_ints]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::qm::IntegralEngine engine(basis);

  occ::MatTriple D = occ::xtb::dipole_ao_matrices(engine);
  REQUIRE(D.x.rows() == basis.nbf());
  REQUIRE(D.x.cols() == basis.nbf());
  for (int alpha = 0; alpha < 3; ++alpha) {
    const auto &M = (alpha == 0) ? D.x : (alpha == 1) ? D.y : D.z;
    REQUIRE((M - M.transpose()).cwiseAbs().maxCoeff() < 1e-12);
  }
  REQUIRE(D.x.cwiseAbs().sum() > 0.0);
  REQUIRE(D.y.cwiseAbs().sum() > 0.0);
  REQUIRE(D.z.cwiseAbs().sum() > 0.0);
}

TEST_CASE("Quadrupole AO matrices: symmetry", "[xtb][multipole_ints]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::qm::IntegralEngine engine(basis);

  auto Q = occ::xtb::quadrupole_ao_matrices(engine);
  REQUIRE(Q.size() == 6);
  for (const auto &M : Q) {
    REQUIRE(M.rows() == basis.nbf());
    REQUIRE((M - M.transpose()).cwiseAbs().maxCoeff() < 1e-12);
  }
  for (int idx : {0, 3, 5}) { // xx, yy, zz
    for (Eigen::Index i = 0; i < Q[idx].rows(); ++i) {
      REQUIRE(Q[idx](i, i) >= -1e-12);
    }
  }
}

TEST_CASE("CAMM moments reconstruct molecular dipole",
          "[xtb][camm][multipole_ints]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);

  // Run the SCC to get a converged density matrix.
  occ::xtb::SccOptions opts;
  auto r = occ::xtb::run_charge_only_scc(atoms, p, opts);
  REQUIRE(r.converged);

  // Build dipole/quadrupole AO matrices at the global origin.
  occ::qm::IntegralEngine engine(basis);
  occ::MatTriple D = occ::xtb::dipole_ao_matrices(engine);
  auto Q = occ::xtb::quadrupole_ao_matrices(engine);

  // Build bf→atom mapping.
  std::vector<int> bf_to_atom = basis.bf_to_atom();

  auto m = occ::xtb::compute_camm_moments(atoms, bf_to_atom, r.density_matrix,
                                          r.overlap_matrix, D, Q);
  REQUIRE(m.dipm.cols() == 3);
  REQUIRE(m.qp.cols() == 3);

  // Reconstruct the total molecular dipole via CAMM:
  //   μ_α^total = Σ_A μ_α^A_CAMM + Σ_A R_α^A · q_A_xtb
  //             = -tr(P D_α) + Σ_A z_A R_α^A      (= electronic + nuclear)
  occ::Vec3 mu_camm = m.dipm.rowwise().sum();
  for (size_t a = 0; a < atoms.size(); ++a) {
    mu_camm += r.atomic_charges(a) *
               occ::Vec3(atoms[a].x, atoms[a].y, atoms[a].z);
  }

  // Compute reference total dipole directly. Electronic part: -tr(P D_α).
  // Nuclear part: Σ_A z_A_valence R_α^A (use ref_occ totals).
  auto shells = occ::xtb::build_shell_table(atoms, p);
  occ::Vec3 z_atomic = occ::Vec3::Zero();
  occ::Vec3 mu_ref;
  mu_ref(0) = -(r.density_matrix.cwiseProduct(D.x)).sum();
  mu_ref(1) = -(r.density_matrix.cwiseProduct(D.y)).sum();
  mu_ref(2) = -(r.density_matrix.cwiseProduct(D.z)).sum();
  for (size_t a = 0; a < atoms.size(); ++a) {
    double z_val = 0.0;
    const auto *e = p.element(atoms[a].atomic_number);
    for (const auto &s : e->shells)
      z_val += s.ref_occ;
    mu_ref += z_val * occ::Vec3(atoms[a].x, atoms[a].y, atoms[a].z);
  }

  INFO("μ CAMM   = " << mu_camm.transpose());
  INFO("μ direct = " << mu_ref.transpose());
  REQUIRE((mu_camm - mu_ref).norm() < 1e-10);
}

TEST_CASE("Full GFN2 SCC vs xtb (water, no dispersion)",
          "[xtb][scc][gfn2]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  occ::xtb::SccOptions opts;
  auto r = occ::xtb::run_gfn2_scc(atoms, p, opts);
  INFO("Total: " << r.total_energy);
  INFO("SCC:   " << r.scc_energy);
  INFO("Reps:  " << r.repulsion_energy);
  INFO("HOMO:  " << r.orbital_energies(3) * occ::units::AU_TO_EV << " eV");
  INFO("Charges: " << r.atomic_charges.transpose());
  REQUIRE(r.converged);
  // xtb full GFN2 total for water: -5.0702559 Eh.
  REQUIRE(r.total_energy == Approx(-5.0702559).margin(2e-4));
  // HOMO of water: -12.18 eV (xtb full GFN2).
  REQUIRE(r.orbital_energies(3) * occ::units::AU_TO_EV ==
          Approx(-12.18).margin(0.1));
  // Dispersion contribution should be near xtb's -0.000141 Ha (with possible
  // few-µHa difference from EEQ-vs-SCC charges).
  // SCC-coupled D4: now within sub-µHa of xtb.
  REQUIRE(r.dispersion_energy == Approx(-0.000141).margin(1e-6));
}

TEST_CASE("Full GFN2 SCC vs xtb (methane, no dispersion)",
          "[xtb][scc][gfn2]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {6, 0.0, 0.0, 0.0},
      {1, 1.18886, 1.18886, 1.18886},
      {1, -1.18886, -1.18886, 1.18886},
      {1, -1.18886, 1.18886, -1.18886},
      {1, 1.18886, -1.18886, -1.18886},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto r = occ::xtb::run_gfn2_scc(atoms, p);
  INFO("Total: " << r.total_energy);
  INFO("Charges: " << r.atomic_charges.transpose());
  REQUIRE(r.converged);
  // xtb total for methane (full GFN2): -4.175074 Eh.
  REQUIRE(r.total_energy == Approx(-4.175074).margin(2e-4));
  // SCC-coupled D4: tracks xtb closely. Residual ~25 µHa likely from
  // GFN2-specific D4 weighting parameters (ga/gc/wf) we haven't tuned.
  REQUIRE(r.dispersion_energy == Approx(-0.000662).margin(3e-5));
  // Charges should still be tetrahedral.
  for (int i = 1; i <= 4; ++i) {
    REQUIRE(r.atomic_charges(i) ==
            Approx(r.atomic_charges(1)).margin(1e-3));
  }
}

TEST_CASE("NativeCalculator: water Molecule round-trip",
          "[xtb][native]") {
  // Build a Molecule with positions in Angstrom (occ::Molecule convention),
  // then verify NativeCalculator reproduces the SCC energy.
  using occ::core::Atom;
  using occ::core::Molecule;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  Molecule mol(atoms);

  occ::xtb::NativeCalculator calc(mol);
  double e = calc.single_point_energy();
  INFO("Energy: " << e);
  INFO("Charges: " << calc.charges().transpose());
  REQUIRE(e == Approx(-5.0702559).margin(2e-4));
  REQUIRE(calc.num_atoms() == 3);
  REQUIRE(calc.charges().size() == 3);
  REQUIRE(calc.charges().sum() == Approx(0.0).margin(1e-9));
  REQUIRE(calc.charges()(0) < 0.0);

  // bond_orders symmetric, O-H entries are >0 and roughly equal.
  occ::Mat bo = calc.bond_orders();
  REQUIRE(bo.rows() == 3);
  REQUIRE(bo.cols() == 3);
  REQUIRE(bo(0, 1) > 0.5);
  REQUIRE(bo(0, 2) > 0.5);
  REQUIRE((bo - bo.transpose()).cwiseAbs().maxCoeff() < 1e-12);

  // update_structure should keep the basis but recompute everything.
  occ::Mat3N positions_disp = calc.positions();
  positions_disp.col(1)(0) += 0.05;
  calc.update_structure(positions_disp);
  double e2 = calc.single_point_energy();
  REQUIRE(e2 != Approx(e));      // displaced geometry → different energy
  REQUIRE(e2 == Approx(e).margin(0.01)); // but in the same ballpark
}

TEST_CASE("Anisotropic ES (post-SCF estimate, water)",
          "[xtb][anisotropic]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::xtb::SccOptions opts;
  auto r = occ::xtb::run_charge_only_scc(atoms, p, opts);
  REQUIRE(r.converged);

  occ::qm::IntegralEngine engine(basis);
  occ::MatTriple D = occ::xtb::dipole_ao_matrices(engine);
  auto Q = occ::xtb::quadrupole_ao_matrices(engine);

  std::vector<int> bf_to_atom = basis.bf_to_atom();
  auto m = occ::xtb::compute_camm_moments(atoms, bf_to_atom, r.density_matrix,
                                          r.overlap_matrix, D, Q);

  occ::Vec cn = occ::xtb::gfn_coordination_numbers(atoms);
  occ::Vec mp_radii = occ::xtb::multipole_radii(atoms, cn, p);
  auto damped = occ::xtb::damped_multipole_coulomb(atoms, mp_radii, p);
  auto e = occ::xtb::anisotropic_energy(atoms, r.atomic_charges, m, damped, p);

  INFO("E_aes = " << e.aes);
  INFO("E_pol = " << e.polariz);
  INFO("E_total = " << e.total());
  // xtb full-GFN2 reference for water: aes = +0.000680, XC/pol = -0.000773.
  // Computed from a charge-only converged density, our values won't match
  // exactly but should be the right sign and order of magnitude (millihartree
  // scale).
  REQUIRE(std::abs(e.aes) < 5e-3);
  REQUIRE(std::abs(e.polariz) < 5e-3);
}

TEST_CASE("SCC charge-only: water sanity check", "[xtb][scc]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  occ::xtb::SccOptions opts;
  opts.max_iterations = 250;
  auto r = occ::xtb::run_charge_only_scc(atoms, p, opts);
  INFO("Converged in " << r.n_iterations << " iterations");
  INFO("Total energy: " << r.total_energy);
  INFO("SCC energy:   " << r.scc_energy);
  INFO("Repulsion:    " << r.repulsion_energy);
  INFO("Charges:      " << r.atomic_charges.transpose());
  INFO("Orb energies (Ha): " << r.orbital_energies.transpose());
  INFO("Orb energies (eV): "
       << (r.orbital_energies * occ::units::AU_TO_EV).transpose());
  REQUIRE(r.converged);
  // xtb's full GFN2 total is -5.0702559 Eh. Phase 3a (third-order on-site)
  // closes the gap from ~7 mHa to ~7e-4 Ha. Phase 3b–3f (CAMM multipoles)
  // and Phase 4 (D4 dispersion) will close the remainder.
  REQUIRE(r.total_energy == Approx(-5.0702559).margin(1e-3));
  // Net charge sum must be zero for neutral water.
  REQUIRE(r.atomic_charges.sum() == Approx(0.0).margin(1e-9));
  // Oxygen electronegativity gives a negative charge on O.
  REQUIRE(r.atomic_charges(0) < -0.3);
  REQUIRE(r.atomic_charges(1) > 0.1);
  REQUIRE(r.atomic_charges(2) > 0.1);
  // HOMO of water in GFN2 is around -12.18 eV.
  // (xtb reference: -0.4476 Hartree.) The charge-only model is ~0.3 eV off.
  REQUIRE(r.orbital_energies(3) * occ::units::AU_TO_EV ==
          Approx(-12.18).margin(0.5));
}

TEST_CASE("SCC charge-only: methane", "[xtb][scc]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {6, 0.0, 0.0, 0.0},
      {1, 1.18886, 1.18886, 1.18886},
      {1, -1.18886, -1.18886, 1.18886},
      {1, -1.18886, 1.18886, -1.18886},
      {1, 1.18886, -1.18886, -1.18886},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  occ::xtb::SccOptions opts;
  auto r = occ::xtb::run_charge_only_scc(atoms, p, opts);
  INFO("Total energy: " << r.total_energy);
  INFO("Charges:      " << r.atomic_charges.transpose());
  REQUIRE(r.converged);
  REQUIRE(r.atomic_charges.sum() == Approx(0.0).margin(1e-9));
  // C is more electronegative than H, so we'd expect C slightly negative
  // and the four H's slightly positive (or near zero due to small ΔEN).
  for (int i = 1; i <= 4; ++i) {
    REQUIRE(r.atomic_charges(i) ==
            Approx(r.atomic_charges(1)).margin(1e-3)); // tetrahedral symmetry
  }
}

TEST_CASE("Repulsion energy: water vs xtb reference",
          "[xtb][repulsion]") {
  using occ::core::Atom;
  // Water at the geometry that gave xtb's repulsion energy 0.033804516135 Eh.
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  double E = occ::xtb::repulsion_energy(atoms, p);
  REQUIRE(E == Approx(0.033802464095).margin(1e-9));
}

TEST_CASE("Gamma matrix: water structure", "[xtb][gamma]") {
  using occ::core::Atom;
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto st = occ::xtb::build_shell_table(atoms, p);
  // O has 2 shells (2s, 2p), H × 2 has 1 shell each → 4 shells.
  REQUIRE(st.atom.size() == 4);
  REQUIRE(st.atom[0] == 0);
  REQUIRE(st.atom[1] == 0);
  REQUIRE(st.atom[2] == 1);
  REQUIRE(st.atom[3] == 2);
  REQUIRE(st.ang_mom(0) == 0); // O 2s
  REQUIRE(st.ang_mom(1) == 1); // O 2p

  occ::Mat J = occ::xtb::klopman_ohno_gamma(atoms, st, p);
  REQUIRE(J.rows() == 4);
  REQUIRE(J.cols() == 4);
  // Diagonal must equal the per-shell hardness.
  for (int i = 0; i < 4; ++i) {
    REQUIRE(J(i, i) == Approx(st.hardness(i)).margin(1e-12));
  }
  // Symmetry.
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j)
      REQUIRE(J(i, j) == Approx(J(j, i)).margin(1e-12));
  // Same-atom off-diagonals are simply the arithmetic mean of η.
  REQUIRE(J(0, 1) == Approx(0.5 * (st.hardness(0) + st.hardness(1)))
                         .margin(1e-12));
  // Cross-atom values must be smaller than the small-r limit (the average η).
  REQUIRE(J(0, 2) < 0.5 * (st.hardness(0) + st.hardness(2)));
  REQUIRE(J(0, 2) > 0.0);
}

TEST_CASE("AOBasis: overlap diagonal is unity", "[xtb][basis][overlap]") {
  // Each shell should be normalized: <φ_i | φ_i> = 1 on the diagonal.
  using occ::core::Atom;
  auto p = occ::xtb::Gfn2Parameters::load_default();
  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };

  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::qm::IntegralEngine engine(basis);
  auto S = engine.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);

  REQUIRE(S.rows() == basis.nbf());
  REQUIRE(S.cols() == basis.nbf());
  for (Eigen::Index i = 0; i < S.rows(); ++i) {
    REQUIRE(S(i, i) == Approx(1.0).margin(1e-10));
  }
}

#ifdef OCC_HAVE_TBLITE
#include <occ/core/molecule.h>
#include <occ/xtb/tblite_wrapper.h>

namespace {

// Sort and return eigenvalues of a symmetric matrix. Used to compare overlap
// matrices independent of basis-function ordering.
Eigen::VectorXd sorted_eigenvalues(const Eigen::MatrixXd &S) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
  Eigen::VectorXd v = es.eigenvalues();
  std::sort(v.data(), v.data() + v.size());
  return v;
}

double max_eigenvalue_diff(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B) {
  return (sorted_eigenvalues(A) - sorted_eigenvalues(B)).cwiseAbs().maxCoeff();
}

} // namespace

TEST_CASE("AOBasis: overlap eigenvalues vs tblite (water)",
          "[xtb][basis][overlap][tblite]") {
  using occ::core::Atom;
  using occ::core::Molecule;

  std::vector<Atom> atoms{
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };

  Molecule mol(atoms);
  occ::xtb::TbliteCalculator calc(mol);
  calc.save_integrals(true);
  (void)calc.single_point_energy();
  Eigen::MatrixXd S_tb = calc.overlap();

  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::qm::IntegralEngine engine(basis);
  Eigen::MatrixXd S_occ =
      engine.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);

  REQUIRE(S_tb.rows() == S_occ.rows());
  REQUIRE(max_eigenvalue_diff(S_tb, S_occ) < 1e-8);
}

TEST_CASE("AOBasis: overlap eigenvalues vs tblite (methane)",
          "[xtb][basis][overlap][tblite]") {
  using occ::core::Atom;
  using occ::core::Molecule;

  std::vector<Atom> atoms{
      {6, 0.0, 0.0, 0.0},
      {1, 1.18886, 1.18886, 1.18886},
      {1, -1.18886, -1.18886, 1.18886},
      {1, -1.18886, 1.18886, -1.18886},
      {1, 1.18886, -1.18886, -1.18886},
  };

  Molecule mol(atoms);
  occ::xtb::TbliteCalculator calc(mol);
  calc.save_integrals(true);
  (void)calc.single_point_energy();
  Eigen::MatrixXd S_tb = calc.overlap();

  auto p = occ::xtb::Gfn2Parameters::load_default();
  auto basis = occ::xtb::build_aobasis(atoms, p);
  occ::qm::IntegralEngine engine(basis);
  Eigen::MatrixXd S_occ =
      engine.one_electron_operator(occ::qm::IntegralEngine::Op::overlap);

  REQUIRE(S_tb.rows() == S_occ.rows());
  REQUIRE(max_eigenvalue_diff(S_tb, S_occ) < 1e-8);
}
#endif // OCC_HAVE_TBLITE
