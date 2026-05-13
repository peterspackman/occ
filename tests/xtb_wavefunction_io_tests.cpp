#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <occ/core/atom.h>
#include <occ/core/data_directory.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/qm/io/wavefunction_json.h>
#include <occ/qm/wavefunction.h>
#include <occ/xtb/xtb_calculator.h>
#include <sstream>

#ifndef OCC_GFN2_DATA_DIR
#define OCC_GFN2_DATA_DIR "share"
#endif

using Catch::Approx;

namespace {

struct DataDirGuard {
  DataDirGuard() { occ::set_data_directory(OCC_GFN2_DATA_DIR); }
};
DataDirGuard _guard;

// Water geometry in Bohr — same atoms used throughout xtb_native_tests.cpp.
inline std::vector<occ::core::Atom> water_atoms_bohr() {
  return {
      {8, -1.3269576, -0.1059386, 0.0187882},
      {1, -1.9316642, 1.6001735, -0.0217105},
      {1, 0.4866441, 0.0795981, 0.0098625},
  };
}

// In-memory JSON round-trip — mirrors the existing pattern in
// tests/wavefunction_io_tests.cpp.
occ::qm::Wavefunction round_trip_json(const occ::qm::Wavefunction &wfn) {
  occ::io::JsonWavefunctionWriter writer;
  std::string json = writer.to_string(wfn);
  std::istringstream stream(json);
  occ::io::JsonWavefunctionReader reader(stream);
  return reader.wavefunction();
}

// A small grid of off-atom probe points (Bohr) for evaluating density. Any
// non-zero offsets work; we just need values that depend on every basis
// function, so we sample around the molecule.
occ::Mat3N probe_points() {
  occ::Mat3N pts(3, 5);
  pts.col(0) = occ::Vec3(0.0, 0.0, 0.0);
  pts.col(1) = occ::Vec3(0.5, 0.5, 0.5);
  pts.col(2) = occ::Vec3(-0.5, 0.7, -0.3);
  pts.col(3) = occ::Vec3(1.0, -0.2, 0.4);
  pts.col(4) = occ::Vec3(-1.2, -0.8, 0.6);
  return pts;
}

} // namespace

TEST_CASE("XtbCalculator(Molecule): to_wavefunction round-trips through JSON",
          "[xtb][wavefunction][io]") {
  using occ::core::Molecule;
  auto atoms = water_atoms_bohr();
  Molecule mol(atoms);

  occ::xtb::XtbCalculator calc(mol);
  (void)calc.single_point_energy();
  REQUIRE(calc.last_result().converged);

  occ::qm::Wavefunction wfn = calc.to_wavefunction();

  REQUIRE(wfn.method == "GFN2-xTB");
  REQUIRE(wfn.atoms.size() == 3);
  REQUIRE(wfn.nbf > 0);
  REQUIRE(wfn.mo.C.rows() == static_cast<Eigen::Index>(wfn.nbf));
  REQUIRE(wfn.mo.D.rows() == static_cast<Eigen::Index>(wfn.nbf));
  // GFN2 valence-only: H has 1 valence electron, O has 6 → 8 total for water.
  REQUIRE(wfn.num_electrons == 8);
  REQUIRE(wfn.charge() == 0);

  // xTB extras populated.
  REQUIRE(wfn.have_xtb_data);
  REQUIRE(wfn.xtb_converged);
  REQUIRE(wfn.xtb_n_iterations > 0);
  REQUIRE(wfn.xtb_atomic_charges.size() == 3);
  REQUIRE(std::abs(wfn.xtb_scc_energy + wfn.xtb_repulsion_energy +
                   wfn.xtb_dispersion_energy - wfn.energy.total) < 1e-10);

  auto wfn2 = round_trip_json(wfn);

  REQUIRE(wfn2.method == "GFN2-xTB");
  REQUIRE(wfn2.basis == wfn.basis);
  REQUIRE(wfn2.atoms.size() == wfn.atoms.size());
  REQUIRE(wfn2.nbf == wfn.nbf);
  REQUIRE(wfn2.num_electrons == wfn.num_electrons);
  REQUIRE(wfn2.mo.n_alpha == wfn.mo.n_alpha);
  REQUIRE(wfn2.mo.n_beta == wfn.mo.n_beta);
  // Linear-algebra fields are stored as Eigen matrices; require bit-equality
  // (JSON encodes doubles with full precision).
  REQUIRE((wfn2.mo.C - wfn.mo.C).cwiseAbs().maxCoeff() < 1e-12);
  REQUIRE((wfn2.mo.D - wfn.mo.D).cwiseAbs().maxCoeff() < 1e-12);
  REQUIRE((wfn2.mo.energies - wfn.mo.energies).cwiseAbs().maxCoeff() < 1e-12);

  // xTB extras survive the round-trip.
  REQUIRE(wfn2.have_xtb_data);
  REQUIRE(wfn2.xtb_scc_energy == wfn.xtb_scc_energy);
  REQUIRE(wfn2.xtb_repulsion_energy == wfn.xtb_repulsion_energy);
  REQUIRE(wfn2.xtb_dispersion_energy == wfn.xtb_dispersion_energy);
  REQUIRE(wfn2.xtb_converged == wfn.xtb_converged);
  REQUIRE(wfn2.xtb_n_iterations == wfn.xtb_n_iterations);
  REQUIRE((wfn2.xtb_atomic_charges - wfn.xtb_atomic_charges)
              .cwiseAbs()
              .maxCoeff() < 1e-12);

  // What downstream cube / isosurface tools actually call.
  occ::Mat3N pts = probe_points();
  occ::Vec rho1 = wfn.electron_density(pts);
  occ::Vec rho2 = wfn2.electron_density(pts);
  REQUIRE((rho1 - rho2).cwiseAbs().maxCoeff() < 1e-10);
  // Density is non-negative everywhere and non-trivial somewhere.
  REQUIRE(rho1.minCoeff() >= -1e-12);
  REQUIRE(rho1.maxCoeff() > 1e-3);
}

TEST_CASE("XtbCalculator(Crystal): to_wavefunction returns a Γ-only snapshot",
          "[xtb][wavefunction][io][periodic]") {
  // Same water-in-an-8-Å-cube geometry as the existing periodic test in
  // xtb_native_tests.cpp.
  using occ::crystal::AsymmetricUnit;
  using occ::crystal::Crystal;
  using occ::crystal::SpaceGroup;
  using occ::crystal::UnitCell;

  occ::Mat3N positions_ang(3, 3);
  positions_ang.col(0) = occ::Vec3(1.0, 1.0, 1.0);
  positions_ang.col(1) = occ::Vec3(1.96, 1.0, 1.0);
  positions_ang.col(2) = occ::Vec3(0.76, 1.93, 1.0);
  occ::IVec Z(3);
  Z << 8, 1, 1;
  UnitCell cell = occ::crystal::cubic_cell(8.0);
  AsymmetricUnit asym(cell.to_fractional(positions_ang), Z);
  SpaceGroup sg(1);
  Crystal crystal(asym, sg, cell);

  occ::xtb::XtbCalculator calc(crystal);
  calc.set_include_multipoles(true);
  (void)calc.single_point_energy();
  REQUIRE(calc.last_result().converged);

  // Before the fix this throws (m_calc is unemplaced for the Crystal ctor).
  occ::qm::Wavefunction wfn = calc.to_wavefunction();

  REQUIRE(wfn.method == "GFN2-xTB");
  REQUIRE(wfn.atoms.size() == 3);
  REQUIRE(wfn.nbf > 0);
  REQUIRE(wfn.mo.C.rows() == static_cast<Eigen::Index>(wfn.nbf));
  REQUIRE(wfn.mo.D.rows() == static_cast<Eigen::Index>(wfn.nbf));
  REQUIRE(wfn.num_electrons == 8);

  REQUIRE(wfn.have_xtb_data);
  REQUIRE(wfn.xtb_converged);
  REQUIRE(wfn.xtb_n_iterations > 0);

  auto wfn2 = round_trip_json(wfn);

  REQUIRE(wfn2.basis == wfn.basis);
  REQUIRE(wfn2.atoms.size() == wfn.atoms.size());
  REQUIRE(wfn2.nbf == wfn.nbf);
  REQUIRE((wfn2.mo.C - wfn.mo.C).cwiseAbs().maxCoeff() < 1e-12);
  REQUIRE((wfn2.mo.D - wfn.mo.D).cwiseAbs().maxCoeff() < 1e-12);
  REQUIRE((wfn2.mo.energies - wfn.mo.energies).cwiseAbs().maxCoeff() < 1e-12);

  REQUIRE(wfn2.have_xtb_data);
  REQUIRE(wfn2.xtb_scc_energy == wfn.xtb_scc_energy);
  REQUIRE(wfn2.xtb_repulsion_energy == wfn.xtb_repulsion_energy);
  REQUIRE(wfn2.xtb_dispersion_energy == wfn.xtb_dispersion_energy);
  REQUIRE(wfn2.xtb_converged == wfn.xtb_converged);
  REQUIRE(wfn2.xtb_n_iterations == wfn.xtb_n_iterations);
  REQUIRE((wfn2.xtb_atomic_charges - wfn.xtb_atomic_charges)
              .cwiseAbs()
              .maxCoeff() < 1e-12);

  // Density evaluation works on the central-cell snapshot — that's what
  // makes the periodic GFN2 result consumable by `occ cube` / isosurface.
  // Probe near the central-cell oxygen (Bohr-converted Å positions).
  occ::Mat3N pts(3, 3);
  pts.col(0) =
      occ::Vec3(1.0, 1.0, 1.0) * occ::units::ANGSTROM_TO_BOHR;
  pts.col(1) =
      occ::Vec3(1.5, 1.0, 1.0) * occ::units::ANGSTROM_TO_BOHR;
  pts.col(2) =
      occ::Vec3(1.0, 1.5, 1.0) * occ::units::ANGSTROM_TO_BOHR;
  occ::Vec rho1 = wfn.electron_density(pts);
  occ::Vec rho2 = wfn2.electron_density(pts);
  REQUIRE((rho1 - rho2).cwiseAbs().maxCoeff() < 1e-10);
  REQUIRE(rho1.maxCoeff() > 1e-3);
}
